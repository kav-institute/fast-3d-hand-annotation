"""Original grid/scale candidate generation, with configurable device and batch size."""
import cv2
import numpy as np
import torch
from inference_geometry import _scale_square_no_clip, _crop_square_with_black, make_offsets
from utils.vis import vis_keypoints_np

def _to_model_device(tensor, model):
    try:
        dev = next(model.parameters()).device
    except Exception:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tensor.to(dev)


def _prep_tensor(img_np):
    t = torch.from_numpy(img_np).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def oversample_keypoints_grid_and_scale(
    rgb_img,
    left, top, right, bottom,
    model,
    train_dataset,
    scales=(1.2, 1.3, 1.4, 1.5),
    grid_vals=np.arange(-10, 11, dtype=np.int32),  # −10..10 inclusive
    show_debug=False,
    batch=False,            # set True if model.inference supports batching
    max_batch=128,           # cap batch size
):
    """
    Returns:
      {
        "scales": np.array[S],                     # scales used
        "offsets": np.array[N,2],                  # grid offsets [dx,dy]
        "boxes": np.array[S,N,4],                  # (l,t,r,b) per scale+offset
        "kps_img": np.array[S,N,42,2],             # keypoints back in image coords
        "kps_dicts": list[list[dict]],             # [S][N] dict-of-42
      }
    """
    offsets = make_offsets(grid_vals)  # (N,2)
    S = len(scales)
    N = offsets.shape[0]

    boxes   = np.zeros((S, N, 4), dtype=np.int32)
    kps_img = np.zeros((S, N, 42, 2), dtype=np.float32)

    # Helper to run a batch (list of (l,t,r,b) crops) through the model in chunks
    def run_batch(box_list):
        imgs = []
        meta = []
        # Crop+resize
        for (l, t, r, b) in box_list:
            cropped_rgb, _ = _crop_square_with_black(rgb_img, l, t, r, b)
            cropped_img = cv2.resize(cropped_rgb, (256, 256))
            imgs.append(_prep_tensor(cropped_img))
            meta.append(cropped_img)  # for optional debug visuals

        # Stack into (B,3,256,256)
        batch_tensor = torch.cat(imgs, dim=0)
        batch_tensor = _to_model_device(batch_tensor, model)

        # If your model needs a batch meta_info, adjust here
        meta_info = {"dataset_id": torch.ones((batch_tensor.shape[0],), dtype=torch.int64, device=batch_tensor.device)}

        out = model.inference({"img": batch_tensor}, meta_info=meta_info)

        # Expect out["kps_xy"] as (B,42,2) or (1,42,2) repeating
        kps = out["kps_xy"]
        if isinstance(kps, torch.Tensor):
            kps = kps.detach().cpu().numpy()
        kps = np.asarray(kps, dtype=np.float32)
        if kps.ndim == 2 and kps.shape == (42, 2):
            kps = np.tile(kps[None, ...], (batch_tensor.shape[0], 1, 1))
        if kps.shape[0] != batch_tensor.shape[0] or kps.shape[1:] != (42, 2):
            raise ValueError(f"Expected (B,42,2) keypoints, got {kps.shape}")

        # Normalize handling: if in [0..1], scale to 256
        kmax, kmin = np.max(kps), np.min(kps)
        if (kmax <= 1.001) and (kmin >= -0.001):
            kps_256 = kps * 256.0
        else:
            kps_256 = kps

        return kps_256, meta  # pixel coords in 256x256 space

    for s_idx, sc in enumerate(scales):
        # Base square for this scale
        l0, t0, r0, b0 = _scale_square_no_clip(left, top, right, bottom, scale=float(sc))

        # Build list of shifted boxes for this scale
        shifted_boxes = []
        for (dx, dy) in offsets:
            l = int(l0 + dx); r = int(r0 + dx)
            t = int(t0 + dy); b = int(b0 + dy)
            shifted_boxes.append((l, t, r, b))
        boxes[s_idx] = np.array(shifted_boxes, dtype=np.int32)

        if batch:
            # --- batched path
            i = 0
            while i < N:
                j = min(i + max_batch, N)
                chunk_boxes = shifted_boxes[i:j]
                kps_256, metas = run_batch(chunk_boxes)

                # Map back each crop
                for k, (l, t, r, b) in enumerate(chunk_boxes):
                    side = (b - t)  # square
                    scale = side / 256.0
                    kp = np.empty_like(kps_256[k], dtype=np.float32)
                    kp[:, 0] = l + kps_256[k][:, 0] * scale
                    kp[:, 1] = t + kps_256[k][:, 1] * scale
                    kps_img[s_idx, i + k] = kp

                    if show_debug:
                        score = np.ones((42,), dtype=np.float32)
                        pil_img_1 = vis_keypoints_np(metas[k].copy(), kps_256[k], score, train_dataset.skeleton)
                        cv2.imshow("Cropped+KPs", cv2.cvtColor(np.array(pil_img_1), cv2.COLOR_RGB2BGR))
                        pil_img_2 = vis_keypoints_np(rgb_img.copy(), kp, score, train_dataset.skeleton)
                        dbg = cv2.cvtColor(np.array(pil_img_2), cv2.COLOR_RGB2BGR)
                        cv2.rectangle(dbg, (l, t), (r, b), (0, 255, 0), 1)
                        cv2.imshow("Out (shifted+scaled)", dbg)
                        cv2.waitKey(1)
                i = j
        else:
            # --- simple per-sample path
            for n_idx, (l, t, r, b) in enumerate(shifted_boxes):
                cropped_rgb, _ = _crop_square_with_black(rgb_img, l, t, r, b)
                cropped_img = cv2.resize(cropped_rgb, (256, 256))

                if show_debug:
                    cv2.imshow("Input", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                tensor_img = _prep_tensor(cropped_img)
                tensor_img = _to_model_device(tensor_img, model)
                meta_info = {"dataset_id": torch.from_numpy(np.array(1)).unsqueeze(0).to(tensor_img.device)}
                out = model.inference({"img": tensor_img}, meta_info=meta_info)

                kps = out["kps_xy"]
                if isinstance(kps, torch.Tensor):
                    kps = kps.detach().cpu().numpy()
                kps = np.asarray(kps, dtype=np.float32)
                if kps.ndim == 3 and kps.shape[0] == 1:
                    kps = kps[0]
                if kps.shape != (42, 2):
                    raise ValueError(f"Expected (42,2) keypoints, got {kps.shape}")

                if (kps.max() <= 1.001) and (kps.min() >= -0.001):
                    kps_256 = kps * 256.0
                else:
                    kps_256 = kps

                side = (b - t)
                scale = side / 256.0
                kp = np.empty_like(kps_256, dtype=np.float32)
                kp[:, 0] = l + kps_256[:, 0] * scale
                kp[:, 1] = t + kps_256[:, 1] * scale
                kps_img[s_idx, n_idx] = kp

                if show_debug:
                    score = np.ones((42,), dtype=np.float32)
                    pil_img_1 = vis_keypoints_np(cropped_img.copy(), kps_256, score, train_dataset.skeleton)
                    cv2.imshow("Cropped+KPs", cv2.cvtColor(np.array(pil_img_1), cv2.COLOR_RGB2BGR))
                    pil_img_2 = vis_keypoints_np(rgb_img.copy(), kp, score, train_dataset.skeleton)
                    dbg = cv2.cvtColor(np.array(pil_img_2), cv2.COLOR_RGB2BGR)
                    cv2.rectangle(dbg, (l, t), (r, b), (0, 255, 0), 1)
                    cv2.imshow("Out (shifted+scaled)", dbg)
                    cv2.waitKey(1)

    # Build dicts (optional)
    kps_dicts = [
        [
            {i_kp: (float(x), float(y)) for i_kp, (x, y) in enumerate(kps_img[s, n])}
            for n in range(N)
        ]
        for s in range(S)
    ]

    return {
        "scales": np.asarray(scales, dtype=np.float32),
        "offsets": offsets.astype(np.int32),     # (N,2) [dx,dy]
        "boxes": boxes,                          # (S,N,4) per scale+offset
        "kps_img": kps_img,                      # (S,N,42,2)
        "kps_dicts": kps_dicts,                  # [S][N] dict-of-42
    }

