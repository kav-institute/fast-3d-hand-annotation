"""One real training/validation step through a generated THAB-format fixture."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from make_synthetic_scene import make_scene
HAS_MODEL=all(importlib.util.find_spec(m) for m in ['torch','torchvision','timm','pycocotools'])


@unittest.skipUnless(HAS_MODEL,'Install requirements/detector.txt for training tests')
class TrainingTests(unittest.TestCase):
    def test_one_step_and_validation(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            gt=make_scene(root/'sequence',frames=1)
            cameras=json.loads((root/'sequence/camera_parameters.json').read_text())['cameras']
            c=cameras[0]; K=np.array(c['camera_matrix']);t=np.array(c['t_param']).reshape(3)
            uv=(gt[0]+t)@K.T;uv=uv[:,:2]/uv[:,2:]
            image=root/'sequence/camera_0/000000.jpg';image.parent.mkdir()
            cv2.imwrite(str(image),np.full((480,640,3),128,dtype=np.uint8))
            parents=[-1]+[0 if j%4==1 else j-1 for j in range(1,21)]
            names=[]
            for side,offset in [('r',0),('l',21)]:
                for j,parent in enumerate(parents):
                    # Dataset visualization helpers expect anatomical name suffixes.
                    name=side+'_wrist' if j==0 else side+'_'+['thumb','index','middle','ring','pinky'][(j-1)//4]+str((j-1)%4+1)
                    names.append(f'{name} {offset+j} {parent+offset if parent>=0 else -1}')
            (root/'skeleton_thab.txt').write_text('\n'.join(names)+'\n')
            annotations={'images':[{'id':0,'file_name':'sequence/camera_0/000000.jpg','camera':'camera_0','width':640,'height':480}],
                'annotations':[{'id':0,'image_id':0,'keypoints':uv.tolist(),'keypoints_3d':gt[0].tolist(),'hand_type':'interacting'}]}
            for mode in ['train','val']:
                (root/f'annotations__{mode}.json').write_text(json.dumps(annotations))
            result=subprocess.run([sys.executable,str(ROOT/'detector/src/train.py'),
                '--dataset-root',str(root),'--thab-unit-scale','1','--device','cpu',
                '--epochs','1','--steps-per-epoch','1','--batch-size','2','--workers','0',
                '--max-val-batches','1','--output-dir',str(root/'run')],capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertTrue((root/'run/best_model.pt').is_file())
            self.assertIn('validation=',result.stdout)


if __name__=='__main__':unittest.main()
