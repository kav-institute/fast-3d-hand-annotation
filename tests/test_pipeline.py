"""Geometry and CLI integration checks; no published assets are required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'triangulation/src'))
sys.path.insert(0,str(ROOT/'examples'))
sys.path.insert(0,str(ROOT/'detector/src'))
from make_synthetic_scene import make_scene
from pipeline.dataloading import load_camera_parameters,load_detection_tensor
from run_multiview_triangulation import _load_all_detections
from infer import find_images


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)/'scene'
    def tearDown(self):
        self.tmp.cleanup()
    def run_cli(self,*args):
        return subprocess.run([sys.executable,str(ROOT/'triangulation/src/run_multiview_triangulation.py'),
            '--detection-source-path',str(self.root),'--no-progress',*args],
            capture_output=True,text=True,check=True,timeout=90)
    def test_ant_exact_and_postprocessing(self):
        gt=make_scene(self.root,frames=2)
        self.run_cli('--method','aggregated_ba','--run-post','--post-graph-window','1','--post-graph-overlap','0')
        with np.load(self.root/'outputs_default_aggregated_ba/triangulation_pipeline_outputs.npz') as data:
            self.assertTrue(data['validity'].all())
            self.assertLess(np.max(np.linalg.norm(data['positions_raw']-gt,axis=-1)),.001)
            self.assertIn('positions_graph',data)
            self.assertTrue(np.isfinite(data['positions_graph']).all())
    def test_all_ransac_variants(self):
        gt=make_scene(self.root,frames=1)
        self.run_cli('--method','baseline','candidates_grid','aggregated','soft_marginal',
            '--ransac-iters','12','--ransac-workers','1','--epi-grid-workers','1')
        for method in ['baseline','candidates_grid','aggregated','soft_marginal']:
            with self.subTest(method=method), np.load(self.root/f'outputs_default_{method}/triangulation_pipeline_outputs.npz') as data:
                self.assertTrue(data['validity'].all())
                self.assertLess(np.max(np.linalg.norm(data['positions_raw']-gt,axis=-1)),.01)
    def test_missing_joint_stays_invalid(self):
        make_scene(self.root,frames=2)
        for p in self.root.glob('detections*.npy'):
            a=np.load(p); a[:,:,:,7,3]=0; np.save(p,a)
        self.run_cli('--method','aggregated_ba')
        with np.load(self.root/'outputs_default_aggregated_ba/triangulation_pipeline_outputs.npz') as data:
            self.assertFalse(data['validity'][:,7].any())
            self.assertTrue(np.isnan(data['positions_raw'][:,7]).all())
            self.assertTrue(data['validity'][:,8].all())
    def test_noisy_ant(self):
        gt=make_scene(self.root,frames=2,noise=.5)
        self.run_cli('--method','aggregated_ba')
        with np.load(self.root/'outputs_default_aggregated_ba/triangulation_pipeline_outputs.npz') as data:
            self.assertTrue(data['validity'].all())
            self.assertLess(np.mean(np.linalg.norm(data['positions_raw']-gt,axis=-1)),2.)
    def test_mismatched_frame_counts_rejected(self):
        make_scene(self.root,frames=2)
        p=self.root/'detections_camera_0_2d.npy'; np.save(p,np.load(p)[:1])
        with self.assertRaisesRegex(ValueError,'identical frame/joint'):
            _load_all_detections(self.root,'default')
    def test_invalid_tensor_rejected(self):
        self.root.mkdir()
        p=self.root/'invalid.npy'; np.save(p,np.zeros((2,1,1,42,3)))
        with self.assertRaises(ValueError): load_detection_tensor(p)
        a=np.ones((1,1,1,42,4)); a[...,0]=np.nan; np.save(p,a)
        with self.assertRaisesRegex(ValueError,'finite'): load_detection_tensor(p)
    def test_calibration_consistency(self):
        make_scene(self.root,frames=1)
        p=self.root/'camera_parameters.json'
        d=json.loads(p.read_text()); d['cameras'][0]['t_param'][0][0]+=50
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'projection_matrix'): load_camera_parameters(p)
    def test_frame_names_rejected_before_inference(self):
        for cid,name in [('0','0001.jpg'),('1','0002.jpg')]:
            d=self.root/f'camera_{cid}'; d.mkdir(parents=True); (d/name).touch()
        with self.assertRaisesRegex(ValueError,'Unsynchronized'): find_images(self.root)
    def test_evaluation_handles_entirely_missing_predictions(self):
        from evaluate_pck_json import compute_pck
        gt=np.zeros((2,42,3))
        result=compute_pck(np.full_like(gt,np.nan),np.zeros((2,42),bool),
            gt,np.ones((2,42),bool),[5,10,20],[])
        self.assertEqual(result['overall']['count'],0)
        self.assertIsNone(result['overall']['pck@10'])
    def test_refinement_keeps_unobserved_joints_missing(self):
        make_scene(self.root,frames=2)
        for p in self.root.glob('detections*.npy'):
            a=np.load(p); a[:,:,:,7,3]=0; np.save(p,a)
        self.run_cli('--method','aggregated_ba','--run-post','--post-graph-window','1','--post-graph-overlap','0')
        with np.load(self.root/'outputs_default_aggregated_ba/triangulation_pipeline_outputs.npz') as data:
            self.assertFalse(data['validity'][:,7].any())
            self.assertTrue(np.isnan(data['positions_graph'][:,7]).all())
    def test_camera_id_with_underscore(self):
        make_scene(self.root,frames=1)
        (self.root/'detections_camera_0_2d.npy').rename(self.root/'detections_camera_front_left_2d.npy')
        self.assertIn('front_left',_load_all_detections(self.root,'default'))


if __name__=='__main__': unittest.main()
