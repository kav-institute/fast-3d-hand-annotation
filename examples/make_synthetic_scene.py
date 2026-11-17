#!/usr/bin/env python3
"""Create a tiny calibrated scene without downloading images, models or research data."""
import argparse
import json
from pathlib import Path
import numpy as np


def make_scene(root, frames=12, noise=0.0, seed=7):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    # 42 canonical joints (right then left); all lengths in millimeters.
    hand = np.zeros((21,3), dtype=float)
    hand[:,2] = 800
    for finger in range(5):
        for segment in range(4):
            hand[1+finger*4+segment] = [(finger-2)*18, -(segment+1)*20, 800+finger*3]
    points = np.concatenate([hand+[-85,0,0],hand+[85,0,0]],axis=0)
    positions = np.stack([points + [2*f,0,0] for f in range(frames)])
    K = np.array([[800.,0,320],[0,800,240],[0,0,1]])
    cameras = []
    for cid,center in zip(['0','1','2','3'],[[-180,0,0],[180,0,0],[0,-160,0],[0,160,0]]):
        R = np.eye(3)
        t = -np.asarray(center,dtype=float).reshape(3,1)
        P = K @ np.hstack((R,t))
        cameras.append(dict(camera=cid,camera_matrix=K.tolist(),rotation_matrix=R.tolist(),
            t_param=t.tolist(),projection_matrix=P.tolist(),distortion_coeff=[0.,0.,0.,0.,0.]))
        camera_points = positions + t.ravel()
        uv = camera_points @ K.T
        uv = uv[...,:2]/uv[...,2:]
        candidates = np.ones((frames,1,9,42,4),dtype=np.float32)
        candidates[...,:2] = uv[:,None,None,:,:]+rng.normal(0,noise,(frames,1,9,42,2))
        np.save(root/f'detections_camera_{cid}_2d.npy',candidates)
    (root/'camera_parameters.json').write_text(json.dumps({'cameras':cameras},indent=2)+'\n')
    (root/'gt_keypoints_3d.json').write_text(json.dumps({str(i):dict(world_coord=p.tolist(),joint_valid=[1]*42) for i,p in enumerate(positions)},indent=2)+'\n')
    np.savez_compressed(root/'ground_truth.npz',positions_raw=positions,validity=np.ones((frames,42),dtype=bool))
    return positions


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='New directory (must not exist).')
    p.add_argument('--frames',type=int,default=12)
    p.add_argument('--noise',type=float,default=0.,help='Gaussian candidate noise in pixels.')
    args=p.parse_args()
    if args.frames<1 or args.noise<0:
        p.error('frames must be positive and noise nonnegative')
    make_scene(args.output,args.frames,args.noise)
    print(f'Created synthetic scene in {args.output}')
