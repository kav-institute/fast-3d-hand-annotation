"""Optional CPU model checks, including actual crop-to-triangulation file interoperability."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'detector/src'))
sys.path.insert(0,str(ROOT/'triangulation/src'))
HAS_MODEL=all(importlib.util.find_spec(m) for m in ['torch','torchvision','timm','pycocotools'])


@unittest.skipUnless(HAS_MODEL,'Install requirements/detector.txt for model tests')
class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from model import MultiHandKeypoint
        torch.set_num_threads(2)
        torch.manual_seed(4)
        cls.model=MultiHandKeypoint(21).eval()
    def test_forward_and_crop_coordinates(self):
        import torch
        from candidates_ours import oversample_keypoints_grid_and_scale
        image=np.full((256,256,3),128,dtype=np.uint8)
        with torch.inference_mode():
            result=oversample_keypoints_grid_and_scale(image,0,0,256,256,self.model,None,
                scales=[1.0],grid_vals=np.array([0]),batch=True,max_batch=1)
        self.assertEqual(result['kps_img'].shape,(1,1,42,2))
        self.assertTrue(np.isfinite(result['kps_img']).all())
        self.assertTrue((result['kps_img']>=0).all())
    def test_training_loss_and_backward(self):
        import torch
        self.model.train()
        targets={'joint_coord_untransformed':torch.zeros(2,42,3)}
        targets['joint_coord_untransformed'][...,:2]=128
        meta={'dataset_id':torch.ones(2,dtype=torch.long),
              'intrinsics':{'focal':torch.full((2,2),800.),'princpt':torch.full((2,2),128.)},
              'root_depth_mm_rl':{'right':torch.full((2,1),800.),'left':torch.full((2,1),800.)},
              'inv_trans':torch.tensor([[[1.,0,0],[0,1,0]]]*2)}
        result=self.model({'img':torch.full((2,3,256,256),.5)}, targets=targets,meta_info=meta)
        loss=result['loss_xy']+.3*result['loss_z']+result['loss_shifter']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
    def test_inference_cli_and_tensor_contract(self):
        import torch,cv2,json
        from pipeline.dataloading import load_detection_tensor
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'random-test-model.pt'
            torch.save({'model_state_dict':self.model.state_dict()},checkpoint)
            boxes={}
            for cid in ['0','1']:
                camera=root/f'camera_{cid}';camera.mkdir()
                cv2.imwrite(str(camera/'000000.jpg'),np.full((256,256,3),128,dtype=np.uint8))
                boxes[cid]=[[0,0,256,256]]
            (root/'boxes.json').write_text(json.dumps(boxes))
            subprocess.run([sys.executable,str(ROOT/'detector/src/infer.py'),'--scene',str(root),
                '--checkpoint',str(checkpoint),'--bboxes-json',str(root/'boxes.json'),
                '--device','cpu','--scales','1','--offsets','0','--batch-size','1'],check=True,capture_output=True,text=True,timeout=90)
            for cid in boxes:
                tensor=load_detection_tensor(root/f'detections_camera_{cid}_2d.npy')
                self.assertEqual(tensor.shape,(1,1,42,4))
                self.assertTrue((tensor[...,3]==1).all())
            self.assertTrue((root/'detections_manifest.json').is_file())


if __name__=='__main__':unittest.main()
