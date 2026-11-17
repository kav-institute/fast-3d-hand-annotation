#!/usr/bin/env python3
"""Generate synchronized per-view 2D candidates for the triangulation pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def parse_args(argv=None, default_source='default'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', '--sequence', dest='scene', type=Path, required=True,
                   help='Scene containing camera_<id>/ directories of synchronized images.')
    p.add_argument('--source', choices=['default', 'sapiens', 'mediapipe'], default=default_source)
    p.add_argument('--checkpoint', type=Path, help='Our full state-dict checkpoint or Sapiens TorchScript model.')
    p.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu or cuda:0.')
    p.add_argument('--bboxes-json', type=Path, help='Optional {camera_id: [[left,top,right,bottom], ...]} in frame order; null denotes missing.')
    p.add_argument('--detector-config', type=Path, default=Path(__file__).parent/'data/cascade_rcnn_x101_64x4d_fpn_1class.py')
    p.add_argument('--detector-checkpoint', type=Path, help='Local MMDetection hand detector weights; needed without --bboxes-json.')
    p.add_argument('--image-extension', default='jpg')
    p.add_argument('--scales', type=float, nargs='+', help='Crop scales; defaults: ours 1.2 1.3 1.4 1.5; Sapiens 1.0 1.5.')
    p.add_argument('--offsets', type=int, nargs='+', help='Pixel offsets along each axis (Cartesian product).')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-frames', type=int, help='Process only the first N frames for a smoke test.')
    p.add_argument('--mp-samples', type=int, default=40, help='MediaPipe affine samples; first is unaugmented.')
    p.add_argument('--images-mirrored', action='store_true', help='Input images are mirrored (MediaPipe handedness convention).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--overwrite', action='store_true', help='Replace this source\'s existing detections.')
    args = p.parse_args(argv)
    for key in ['batch_size', 'mp_samples', 'max_frames']:
        if getattr(args, key) is not None and getattr(args, key) < 1:
            p.error('--' + key.replace('_', '-') + ' must be positive')
    if args.scales is not None and any(s <= 0 for s in args.scales):
        p.error('--scales must be positive')
    if args.source != 'mediapipe':
        if args.checkpoint is None or not args.checkpoint.is_file():
            p.error('--checkpoint must point to an existing local model file')
        if args.bboxes_json is None and (args.detector_checkpoint is None or not args.detector_checkpoint.is_file()):
            p.error('provide --bboxes-json or a local --detector-checkpoint')
    return args


def find_images(scene, extension='jpg', max_frames=None):
    """Require identical frame names in all cameras; never silently truncate a stream."""
    cameras = {}
    expected = None
    for directory in sorted(scene.glob('camera_*')):
        if not directory.is_dir():
            continue
        camera_id = directory.name.removeprefix('camera_')
        if not camera_id:
            raise ValueError(f'Empty camera ID: {directory}')
        paths = sorted(directory.glob('*.' + extension.lstrip('.')))
        names = [p.name for p in paths]
        if not names:
            raise ValueError(f'No {extension} images in {directory}')
        if expected is not None and names != expected:
            raise ValueError(f'Unsynchronized frame names in {directory}; every camera must have the same frame names.')
        expected = names
        cameras[camera_id] = paths[:max_frames]
    if len(cameras) < 2:
        raise ValueError(f'Expected at least two camera_<id>/ folders in {scene}')
    return cameras


def read_image(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f'Cannot read image: {path}')
    return img


def load_model(checkpoint, device):
    import torch
    from model import MultiHandKeypoint
    model = MultiHandKeypoint(single_hand_joint_num=21).to(device)
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=True)
    return model.eval()


def generate_learned(args, cameras, outputs):
    import cv2
    import numpy as np
    import torch
    from inference_geometry import _union_bbox_from_mmdet, adapt_mmdet_pipeline, smooth_and_extend_bboxes
    torch.manual_seed(args.seed)
    if args.source == 'default':
        from candidates_ours import oversample_keypoints_grid_and_scale as oversample
        model = load_model(args.checkpoint, args.device)
        scales = args.scales or [1.2, 1.3, 1.4, 1.5]
        offsets = args.offsets if args.offsets is not None else list(range(-10, 11))
    else:
        from candidates_sapiens import oversample_keypoints_grid_and_scale as oversample
        model = torch.jit.load(str(args.checkpoint), map_location=args.device).eval()
        scales = args.scales or [1.0, 1.5]
        offsets = args.offsets if args.offsets is not None else [-9, 0, 9]
    boxes_json = json.loads(args.bboxes_json.read_text()) if args.bboxes_json else None
    detector = None
    if boxes_json is None:
        from mmdet.apis import init_detector
        detector = init_detector(str(args.detector_config), str(args.detector_checkpoint), device=args.device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    with torch.inference_mode():
        for camera_id, paths in cameras.items():
            shape = read_image(paths[0]).shape
            if boxes_json is None:
                boxes = []
                for path in paths:
                    image = read_image(path)
                    if image.shape != shape:
                        raise ValueError(f'Image dimensions changed in camera {camera_id}')
                    boxes.append(_union_bbox_from_mmdet(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detector))
            else:
                boxes = boxes_json.get(camera_id)
                if boxes is None or len(boxes) < len(paths):
                    raise ValueError(f'Missing bounding boxes for camera {camera_id}')
                boxes = boxes[:len(paths)]
                for box in boxes:
                    if box is not None and (len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]):
                        raise ValueError(f'Invalid [left,top,right,bottom] box in camera {camera_id}: {box}')
            boxes = smooth_and_extend_bboxes(boxes, shape, window_size=5)
            result = np.lib.format.open_memmap(outputs[camera_id], mode='w+', dtype=np.float32,
                shape=(len(paths), len(scales), len(offsets)**2, 42, 4))
            result[:] = 0
            for f, (path, box) in enumerate(zip(paths, boxes)):
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                image = read_image(path)
                if image.shape != shape:
                    raise ValueError(f'Image dimensions changed in camera {camera_id}')
                candidates = oversample(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    *box.astype(int), model, None, scales=scales,
                    grid_vals=np.asarray(offsets, dtype=np.int32), batch=True,
                    max_batch=args.batch_size, show_debug=False)['kps_img']
                result[f, ..., :2] = candidates
                # These sources expose coordinates here, not calibrated confidence estimates.
                result[f, ..., 2:] = np.isfinite(candidates).all(axis=-1)[..., None]
            result.flush()
            print(f'Saved {outputs[camera_id]} {result.shape}')
            del result


def generate_mediapipe(args, cameras, outputs):
    import cv2
    import mediapipe as mp
    import numpy as np
    if not hasattr(mp, 'solutions'):
        raise RuntimeError('This adapter requires MediaPipe with the legacy solutions.hands API (see requirements).')
    rng = np.random.default_rng(args.seed)
    for camera_id, paths in cameras.items():
        result = np.lib.format.open_memmap(outputs[camera_id], mode='w+', dtype=np.float32,
            shape=(len(paths), 1, args.mp_samples, 42, 4))
        result[:] = 0
        shape = read_image(paths[0]).shape
        h, w = shape[:2]
        for sample in range(args.mp_samples):
            scale = 1.0 if sample == 0 else rng.uniform(.95, 1.05)
            shift = np.zeros(2) if sample == 0 else rng.uniform(-.05, .05, size=2)*[w,h]
            affine = np.array([[scale,0,(1-scale)*w/2+shift[0]], [0,scale,(1-scale)*h/2+shift[1]]], dtype=np.float32)
            inverse = cv2.invertAffineTransform(affine)
            with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2) as model:
                for f, path in enumerate(paths):
                    image = read_image(path)
                    if image.shape != shape:
                        raise ValueError(f'Image dimensions changed in camera {camera_id}')
                    augmented = cv2.warpAffine(image, affine, (w,h), borderMode=cv2.BORDER_REPLICATE)
                    detection = model.process(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
                    if not detection.multi_hand_landmarks:
                        continue
                    for landmarks, handedness in zip(detection.multi_hand_landmarks, detection.multi_handedness):
                        classification = handedness.classification[0]
                        right = classification.label == 'Right'
                        if not args.images_mirrored:
                            right = not right
                        start = 0 if right else 21
                        # If handedness collides, retain the more confident detection.
                        if result[f,0,sample,start,2] > classification.score:
                            continue
                        uv = np.array([[lm.x*w,lm.y*h] for lm in landmarks.landmark], dtype=np.float32)
                        xy = cv2.transform(uv[None], inverse)[0]
                        result[f,0,sample,start:start+21,:2] = xy
                        result[f,0,sample,start:start+21,2] = classification.score
                        result[f,0,sample,start:start+21,3] = 1
        result.flush()
        print(f'Saved {outputs[camera_id]} {result.shape}')
        del result


def main(argv=None, default_source='default'):
    args = parse_args(argv, default_source)
    cameras = find_images(args.scene, args.image_extension, args.max_frames)
    prefix = {'default':'detections', 'sapiens':'detections_sp', 'mediapipe':'detections_mp'}[args.source]
    outputs = {cid: args.scene/f'{prefix}_camera_{cid}_2d.npy' for cid in cameras}
    existing = [p for p in outputs.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f'Detections already exist; use --overwrite to replace: {existing}')
    if args.source == 'mediapipe':
        generate_mediapipe(args, cameras, outputs)
    else:
        generate_learned(args, cameras, outputs)
    metadata = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    metadata['frames'] = {cid:[p.name for p in paths] for cid,paths in cameras.items()}
    (args.scene/f'{prefix}_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
