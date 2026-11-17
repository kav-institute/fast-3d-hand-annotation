#!/usr/bin/env python3
"""Optional webcam preview of the learned detector (a centered hand crop, no triangulation)."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--camera', type=int, default=0)
    args = p.parse_args()
    import cv2
    import torch
    from infer import load_model
    model = load_model(args.checkpoint, args.device)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')
    try:
        with torch.inference_mode():
            while True:
                ok, image = camera.read()
                if not ok:
                    break
                h,w = image.shape[:2]
                side = min(h,w)
                x,y = (w-side)//2,(h-side)//2
                crop = cv2.resize(image[y:y+side,x:x+side], (256,256))
                tensor = torch.from_numpy(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()[None].to(args.device)/255
                result = model.inference({'img':tensor}, meta_info={'dataset_id':torch.ones(1,dtype=torch.long,device=args.device)})
                for u,v in result['kps_xy'][0].cpu().numpy():
                    cv2.circle(crop,(int(u),int(v)),2,(0,255,0),-1)
                cv2.imshow('Hand keypoints (Esc / q to quit)',cv2.resize(crop,(768,768)))
                if cv2.waitKey(1)&255 in (27,ord('q')):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
