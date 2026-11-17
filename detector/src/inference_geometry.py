"""Crop geometry and bounding-box helpers for candidate generation."""
from __future__ import annotations
import cv2
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmengine import Config, ConfigDict
    from mmengine.structures import InstanceData

def _scale_square_no_clip(left, top, right, bottom, scale=1.3):
    """
    Return the intended *square* bbox after scaling by `scale`, WITHOUT clipping to the image.
    Output: (l, t, r, b) integers; r-l == b-t
    """
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    w = (right - left)
    h = (bottom - top)

    # scale and force square by larger side
    side = max(w * scale, h * scale)

    l = int(round(cx - side / 2.0))
    t = int(round(cy - side / 2.0))
    r = int(round(cx + side / 2.0))
    b = int(round(cy + side / 2.0))

    # re-square with integer math
    side = max(2, r - l, b - t)
    r = l + side
    b = t + side
    return l, t, r, b  # can be outside image bounds


def _crop_square_with_black(rgb_img, l, t, r, b):
    """
    Crop the region [t:b, l:r] from rgb_img; if it goes outside the image,
    pad with black so the returned crop is exactly (side, side, 3).
    Returns: cropped_rgb (np.ndarray), side (int)
    """
    H, W = rgb_img.shape[:2]
    side = r - l  # square side

    # Destination (black) canvas
    crop = np.zeros((side, side, 3), dtype=rgb_img.dtype)

    # Overlap with image
    src_l = max(0, l)
    src_t = max(0, t)
    src_r = min(W, r)
    src_b = min(H, b)

    if src_r <= src_l or src_b <= src_t:
        # Entirely outside -> return black square
        return crop, side

    # Where to paste inside the black canvas
    dst_l = src_l - l
    dst_t = src_t - t
    dst_r = dst_l + (src_r - src_l)
    dst_b = dst_t + (src_b - src_t)

    crop[dst_t:dst_b, dst_l:dst_r] = rgb_img[src_t:src_b, src_l:src_r]
    return crop, side


def smooth_and_extend_bboxes(
    bboxes,
    img_shape,
    area_small_frac=0.6,
    window_size=5,
):
    """
    bboxes: list of (l, t, r, b) or None, length T
    img_shape: (H, W, C) or (H, W)
    area_small_frac: threshold to decide case 2 (one-hand) vs case 1
    window_size: temporal smoothing window (>=1)
    returns: np.ndarray (T, 4) float32
    """
    H, W = img_shape[:2]
    T = len(bboxes)

    arr = np.full((T, 4), np.nan, dtype=np.float32)
    for i, bb in enumerate(bboxes):
        if bb is not None:
            arr[i] = np.array(bb, dtype=np.float32)

    # if everything is missing, just return zeros
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr, dtype=np.float32)

    # ----- forward & backward fill (handles case 3) -----
    last_valid = None
    for i in range(T):
        if not np.any(np.isnan(arr[i])):
            last_valid = arr[i].copy()
        elif last_valid is not None:
            arr[i] = last_valid

    next_valid = None
    for i in reversed(range(T)):
        if not np.any(np.isnan(arr[i])):
            next_valid = arr[i].copy()
        elif next_valid is not None:
            arr[i] = next_valid

    # ----- compute "typical" two-hand box size from stats (mostly case 1) -----
    widths  = arr[:, 2] - arr[:, 0]
    heights = arr[:, 3] - arr[:, 1]

    med_w    = np.median(widths)
    med_h    = np.median(heights)
    med_area = med_w * med_h + 1e-6  # avoid 0

    # ----- extend small boxes (case 2) -----
    out = arr.copy()
    for i in range(T):
        l, t, r, b = out[i]
        w = r - l
        h = b - t
        area = w * h

        # case 2: likely only one hand detected
        if area < area_small_frac * med_area:
            cx = (l + r) * 0.5
            cy = (t + b) * 0.5

            w = med_w
            h = med_h

            l = cx - w * 0.5
            r = cx + w * 0.5
            t = cy - h * 0.5
            b = cy + h * 0.5

            # clamp to image boundaries
            l = max(0, min(l, W - 1))
            r = max(0, min(r, W - 1))
            t = max(0, min(t, H - 1))
            b = max(0, min(b, H - 1))

            out[i] = np.array([l, t, r, b], dtype=np.float32)

    # ----- optional temporal smoothing (handles 1,2,3; mitigates 4) -----
    if window_size > 1:
        half = window_size // 2
        smoothed = np.zeros_like(out)
        for t in range(T):
            start = max(0, t - half)
            end   = min(T, t + half + 1)
            smoothed[t] = out[start:end].mean(axis=0)
        out = smoothed

    return out


def make_offsets(vals=np.arange(-10, 11, dtype=np.int32)):
    # (441, 2) with columns [dx, dy], center [0, 0]
    X, Y = np.meshgrid(vals, vals, indexing='xy')
    return np.column_stack([X.ravel(), Y.ravel()])  # (441, 2)


def adapt_mmdet_pipeline(cfg: Config | ConfigDict) -> Config | ConfigDict:
    """Converts pipeline types in MMDetection's test dataloader to use the
    'mmdet' namespace.

    Args:
        cfg (ConfigDict): Configuration dictionary for MMDetection.

    Returns:
        ConfigDict: Configuration dictionary with updated pipeline types.
    """
    # use lazy import to avoid hard dependence on mmdet
    from mmdet.datasets import transforms

    if 'test_dataloader' not in cfg:
        return cfg

    pipeline = cfg.test_dataloader.dataset.pipeline
    for trans in pipeline:
        if trans['type'] in dir(transforms):
            trans['type'] = 'mmdet.' + trans['type']

    return cfg


def _union_bbox_from_mmdet(
    rgb_img,
    detector,
    det_cat_id=0,
    scales=(0.5, 0.7, 1.0),
    flips=("none", "h", "v", "hv"),
    bbox_thr=0.1,
    max_hands=2,
):
    """
    Return union bbox (left, top, right, bottom) over all hands detected by MMDetection,
    using multi-scale + flipping augmentation.
    """
    from mmdet.apis import inference_detector
    # MMDetection expects an OpenCV BGR array.
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    H, W = rgb_img.shape[:2]
    all_boxes = []

    # --------------------------------------------------------
    # Helper to transform detected bboxes back to original
    # --------------------------------------------------------
    def unflip_bboxes(bboxes, flip_type, scale):
        """Undo flip & scaling to original image coordinates."""
        b = bboxes.copy()

        # Undo scale
        b /= scale

        if flip_type == "h":
            # horizontal flip: x -> W - x
            b[:, [0, 2]] = W - b[:, [2, 0]]

        elif flip_type == "v":
            # vertical flip: y -> H - y
            b[:, [1, 3]] = H - b[:, [3, 1]]

        elif flip_type == "hv":
            # both flips
            b[:, [0, 2]] = W - b[:, [2, 0]]
            b[:, [1, 3]] = H - b[:, [3, 1]]

        return b

    # --------------------------------------------------------
    # Main multiscale + flip loop
    # --------------------------------------------------------
    for flip in flips:
        # Prepare flipped version
        if flip == "none":
            base_img = rgb_img
        elif flip == "h":
            base_img = cv2.flip(rgb_img, 1)
        elif flip == "v":
            base_img = cv2.flip(rgb_img, 0)
        elif flip == "hv":
            base_img = cv2.flip(rgb_img, -1)
        else:
            raise ValueError(f"Unknown flip type: {flip}")

        for s in scales:
            resized = cv2.resize(base_img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

            # Detection
            result = inference_detector(detector, resized)
            pred: InstanceData = result.pred_instances.cpu().numpy()

            if len(pred) == 0 or len(pred.bboxes) == 0:
                continue

            bboxes_s = pred.bboxes
            scores_s = pred.scores

            # Filter by class
            if pred.labels is not None:
                idx = np.where(pred.labels == det_cat_id)[0]
                bboxes_s = bboxes_s[idx]
                scores_s = scores_s[idx]

            # Threshold
            keep = scores_s > bbox_thr
            bboxes_s = bboxes_s[keep]
            scores_s = scores_s[keep]

            if len(bboxes_s) == 0:
                continue

            # Undo flip + undo scaling
            bboxes_orig = unflip_bboxes(bboxes_s, flip, scale=s)

            all_boxes.append(bboxes_orig)

    # --------------------------------------------------------
    # If nothing detected
    # --------------------------------------------------------
    if len(all_boxes) == 0:
        return None

    # Combine boxes from the selected scales and flips.
    all_b = np.concatenate(all_boxes, axis=0)

    if len(all_b) == 0:
        return None

    # --------------------------------------------------------
    # Compute union box
    # --------------------------------------------------------
    x_min = np.min(all_b[:, 0])
    y_min = np.min(all_b[:, 1])
    x_max = np.max(all_b[:, 2])
    y_max = np.max(all_b[:, 3])

    # Clamp
    x_min, x_max = np.clip([x_min, x_max], 0, W - 1)
    y_min, y_max = np.clip([y_min, y_max], 0, H - 1)

    return int(x_min), int(y_min), int(x_max), int(y_max)

