"""Sapiens TorchScript inference over scaled and shifted image crops."""
from __future__ import annotations
from typing import List
import cv2
import numpy as np
import torch
from inference_geometry import _scale_square_no_clip, _crop_square_with_black, make_offsets
from utils.vis import vis_keypoints_np

NUM_KPTS_WHOLEBODY = 133

# COCO WholeBody hand indices (standard ordering)
LEFT_HAND_IDXS = list(range(91, 112))  # 21 keypoints
RIGHT_HAND_IDXS = list(range(112, 133))   # 21 keypoints

# Sapiens input spec (matches the demo script: shape [1024, 768])
SAPIENS_INPUT_H = 1024
SAPIENS_INPUT_W = 768
SAPIENS_INPUT_SHAPE = (SAPIENS_INPUT_W, SAPIENS_INPUT_H)  # (W, H) like in their code
SAPIENS_HEATMAP_SCALE = 4  # same as args.heatmap_scale in the demo

SAPIENS_MEAN = np.array([123.5, 116.5, 103.5], dtype=np.float32)
SAPIENS_STD  = np.array([58.5, 57.0, 57.5], dtype=np.float32)

def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def get_heatmap_maximum(heatmaps: np.ndarray):
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
                              blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian,
                                  derivative).squeeze()

    return keypoints


def udp_decode(heatmaps, input_size, heatmap_size, blur_kernel_size=11) -> np.ndarray:
    """UDP decoding for keypoint location refinement.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        blur_kernel_size (int): Gaussian kernel size (K) for modulation, which
            should match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Refined keypoint locations.
    """
    keypoints, scores = get_heatmap_maximum(heatmaps)
    # unsqueeze the instance dimension for single-instance results
    keypoints = keypoints[None]
    scores = scores[None]
    keypoints = refine_keypoints_dark_udp(
        keypoints, heatmaps, blur_kernel_size=blur_kernel_size)

    W, H = heatmap_size
    keypoints = (keypoints / [W - 1, H - 1]) * input_size
    return keypoints, scores


def _sapiens_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sapiens_preprocess_batch(patches_rgb: List[np.ndarray]) -> torch.Tensor:
    """
    patches_rgb: list of np.ndarray (H,W,3), RGB [0..255]
    Returns:
        imgs: torch.Tensor (B,3,H_in,W_in), normalized as in Meta's script.
    """
    imgs = []
    mean_t = torch.from_numpy(SAPIENS_MEAN).view(-1, 1, 1)
    std_t  = torch.from_numpy(SAPIENS_STD).view(-1, 1, 1)

    for patch_rgb in patches_rgb:
        # Sapiens pipeline assumes BGR→RGB inside; we start from RGB, so convert to BGR first.
        patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

        # Resize to model input (W,H)
        img = cv2.resize(
            patch_bgr,
            (SAPIENS_INPUT_W, SAPIENS_INPUT_H),
            interpolation=cv2.INTER_LINEAR,
        )
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = torch.from_numpy(img)   # (3,H,W), BGR

        # BGR -> RGB (same as demo: img[[2,1,0]])
        img = img[[2, 1, 0], ...].float()

        img = (img - mean_t) / std_t
        imgs.append(img)

    return torch.stack(imgs, dim=0)  # (B,3,H_in,W_in)


def _sapiens_infer_patches(
    model,
    patches_rgb: List[np.ndarray],
) -> np.ndarray:
    """
    Run Sapiens on a list of 256x256 RGB patches and return hand keypoints
    in *patch 256x256 coordinates*.

    Returns:
        kps_256: (B, 42, 2) float32
    """
    if len(patches_rgb) == 0:
        return np.zeros((0, 42, 2), dtype=np.float32)

    imgs = _sapiens_preprocess_batch(patches_rgb)  # (B,3,H_in,W_in)
    device = _sapiens_device(model)
    imgs = imgs.to(device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        heatmaps = model(imgs)  # (B, 133, H_hm, W_hm)

    heatmaps = heatmaps.cpu()
    B = heatmaps.shape[0]
    kps_256 = np.zeros((B, 42, 2), dtype=np.float32)

    input_shape = SAPIENS_INPUT_SHAPE  # (W,H), as in the demo

    for i in range(B):
        hm = heatmaps[i]  # (133, H_hm, W_hm)

        # udp_decode expects (num_kpts, H_hm, W_hm)
        heatmap_np = hm.numpy()

        keypoints, keypoint_scores = udp_decode(
            heatmap_np,
            input_shape,
            (
                int(input_shape[0] / SAPIENS_HEATMAP_SCALE),
                int(input_shape[1] / SAPIENS_HEATMAP_SCALE),
            ),
        )  # keypoints: (1,133,2), scores: (1,133)

        keypoints = keypoints[0]  # (133,2)

        # Map from model input size (W,H) -> canonical 256x256 patch coordinates.
        # We treat the whole patch as a 256x256 square.
        scale_patch = np.array([256.0, 256.0], dtype=np.float32)
        wh = np.array(input_shape, dtype=np.float32)  # (W,H)
        kpts_patch = (keypoints / wh) * scale_patch   # (133,2) in 0..256

        # Slice out the 42 hand points: left 21 + right 21
        left_hand  = kpts_patch[LEFT_HAND_IDXS]   # (21,2)
        right_hand = kpts_patch[RIGHT_HAND_IDXS]  # (21,2)
        hands_42   = np.concatenate([right_hand, left_hand], axis=0)  # (42,2)

        kps_256[i] = hands_42.astype(np.float32)

    return kps_256


def oversample_keypoints_grid_and_scale(
    rgb_img,
    left, top, right, bottom,
    model,
    train_dataset,
    scales=(1.0, 1.5),
    grid_vals=np.arange(-9, 10, 9, dtype=np.int32),  # −10..10 inclusive
    show_debug=False,
    batch=True,             # we *use* batch path for Sapiens
    max_batch=64,
):
    """
    Returns:
      {
        "scales":   np.array[S],                     # scales used
        "offsets":  np.array[N,2],                  # grid offsets [dx,dy]
        "boxes":    np.array[S,N,4],                # (l,t,r,b) per scale+offset
        "kps_img":  np.array[S,N,42,2],             # keypoints in *image* coords
        "kps_dicts": list[list[dict]],              # [S][N] dict-of-42
      }
    """
    offsets = make_offsets(grid_vals)  # (N,2)
    S = len(scales)
    N = offsets.shape[0]

    boxes   = np.zeros((S, N, 4), dtype=np.int32)
    kps_img = np.zeros((S, N, 42, 2), dtype=np.float32)

    for s_idx, sc in enumerate(scales):
        # Base square for this scale
        l0, t0, r0, b0 = _scale_square_no_clip(left, top, right, bottom, scale=float(sc))

        shifted_boxes = []
        for (dx, dy) in offsets:
            l = int(l0 + dx)
            r = int(r0 + dx)
            t = int(t0 + dy)
            b = int(b0 + dy)
            shifted_boxes.append((l, t, r, b))
        boxes[s_idx] = np.array(shifted_boxes, dtype=np.int32)

        if batch:
            # --- batched path for Sapiens ---
            i = 0
            while i < N:
                j = min(i + max_batch, N)
                chunk_boxes = shifted_boxes[i:j]

                # Crop & resize patches to 256x256 RGB
                patches = []
                metas   = []
                for (l, t, r, b) in chunk_boxes:
                    cropped_rgb, _ = _crop_square_with_black(rgb_img, l, t, r, b)
                    cropped_img = cv2.resize(cropped_rgb, (256, 256))
                    patches.append(cropped_img)
                    metas.append(cropped_img.copy())

                # Run Sapiens on this batch of patches
                kps_256 = _sapiens_infer_patches(model, patches)  # (B,42,2)

                # Map back each crop's 256x256 coords into the original image
                for k, (l, t, r, b) in enumerate(chunk_boxes):
                    side = (b - t)  # square side in original image
                    scale_box = side / 256.0

                    kp_img = np.empty_like(kps_256[k], dtype=np.float32)
                    kp_img[:, 0] = l + kps_256[k][:, 0] * scale_box
                    kp_img[:, 1] = t + kps_256[k][:, 1] * scale_box
                    kps_img[s_idx, i + k] = kp_img

                    if show_debug:
                        score = np.ones((42,), dtype=np.float32)
                        pil_crop = vis_keypoints_np(metas[k].copy(), kps_256[k], score, train_dataset.skeleton)
                        cv2.imshow("Cropped+Sapiens KPs", cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR))

                        pil_full = vis_keypoints_np(rgb_img.copy(), kp_img, score, train_dataset.skeleton)
                        dbg = cv2.cvtColor(np.array(pil_full), cv2.COLOR_RGB2BGR)
                        cv2.rectangle(dbg, (l, t), (r, b), (0, 255, 0), 1)
                        cv2.imshow("Out (shifted+scaled)", dbg)
                        cv2.waitKey(1)

                i = j

        else:
            # --- per-sample path (no real need, but kept for completeness) ---
            for n_idx, (l, t, r, b) in enumerate(shifted_boxes):
                cropped_rgb, _ = _crop_square_with_black(rgb_img, l, t, r, b)
                cropped_img = cv2.resize(cropped_rgb, (256, 256))

                kps_256 = _sapiens_infer_patches(model, [cropped_img])  # (1,42,2)
                kps_256 = kps_256[0]

                side = (b - t)
                scale_box = side / 256.0

                kp_img = np.empty_like(kps_256, dtype=np.float32)
                kp_img[:, 0] = l + kps_256[:, 0] * scale_box
                kp_img[:, 1] = t + kps_256[:, 1] * scale_box
                kps_img[s_idx, n_idx] = kp_img

                if show_debug:
                    score = np.ones((42,), dtype=np.float32)
                    pil_crop = vis_keypoints_np(cropped_img.copy(), kps_256, score, train_dataset.skeleton)
                    cv2.imshow("Cropped+Sapiens KPs", cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR))
                    pil_full = vis_keypoints_np(rgb_img.copy(), kp_img, score, train_dataset.skeleton)
                    dbg = cv2.cvtColor(np.array(pil_full), cv2.COLOR_RGB2BGR)
                    cv2.rectangle(dbg, (l, t), (r, b), (0, 255, 0), 1)
                    cv2.imshow("Out (shifted+scaled)", dbg)
                    cv2.waitKey(1)

    # Build dicts (like before)
    kps_dicts = [
        [
            {i_kp: (float(x), float(y)) for i_kp, (x, y) in enumerate(kps_img[s, n])}
            for n in range(N)
        ]
        for s in range(S)
    ]

    return {
        "scales":  np.asarray(scales, dtype=np.float32),
        "offsets": offsets.astype(np.int32),  # (N,2)
        "boxes":   boxes,                     # (S,N,4)
        "kps_img": kps_img,                   # (S,N,42,2) in *image* coords
        "kps_dicts": kps_dicts,
    }

