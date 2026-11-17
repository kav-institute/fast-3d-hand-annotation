"""
Camera calibration and detection tensor loading utilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class CameraModel:
    camera_id: str
    K: np.ndarray
    dist_coeff: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


def load_camera_parameters(param_path: Path) -> Dict[str, CameraModel]:
    """Parse the provided JSON calibration into CameraModel objects keyed by camera id."""
    with param_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cameras: Dict[str, CameraModel] = {}
    for cam in data["cameras"]:
        camera_id = str(cam["camera"])
        K = np.array(cam["camera_matrix"], dtype=np.float64)
        dist = np.array(cam["distortion_coeff"], dtype=np.float64).reshape(-1)
        R = np.array(cam["rotation_matrix"], dtype=np.float64)
        t = np.array(cam["t_param"], dtype=np.float64).reshape(3, 1)
        P = np.array(cam["projection_matrix"], dtype=np.float64)
        if camera_id in cameras:
            raise ValueError(f"Duplicate camera ID: {camera_id}")
        if K.shape != (3,3) or R.shape != (3,3) or P.shape != (3,4):
            raise ValueError(f"Invalid calibration matrix shape for camera {camera_id}")
        if not all(np.isfinite(a).all() for a in (K, dist, R, t, P)):
            raise ValueError(f"Non-finite calibration for camera {camera_id}")
        expected = K @ np.hstack((R,t))
        scale = np.sum(P*expected)/np.sum(expected*expected)
        if abs(scale) < 1e-12 or not np.allclose(P, scale*expected, rtol=1e-5, atol=1e-4):
            raise ValueError(f"projection_matrix must be proportional to K [R|t] for camera {camera_id}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R),1,atol=1e-4):
            raise ValueError(f"Invalid rotation matrix for camera {camera_id}")
        if abs(np.linalg.det(K)) < 1e-12:
            raise ValueError(f"Singular intrinsics for camera {camera_id}")
        cameras[camera_id] = CameraModel(camera_id, K, dist, R, t, P)
    return cameras


def load_detection_tensor(det_path: Path, top_k: Optional[int] = None) -> np.ndarray:
    """
    Load detector output and flatten scale/offset hypotheses into a candidate dimension.
    Returns array of shape [F, H, J, 4] (x, y, score, valid).
    """
    raw = np.load(det_path, allow_pickle=False)
    if raw.ndim != 5 or raw.shape[-1] != 4 or any(dim == 0 for dim in raw.shape):
        raise ValueError(f"Unexpected detector tensor shape {raw.shape} in {det_path}")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"Detection tensor must be numeric: {det_path}")
    marked_valid = raw[..., 3] > 0.5
    if not np.isfinite(raw[..., 3]).all() or not np.isfinite(raw[..., :3][marked_valid]).all():
        raise ValueError(f"Valid detections must contain finite x, y and score: {det_path}")
    frames, n_scale, n_offsets, n_joints, _ = raw.shape
    flat = raw.reshape(frames, n_scale * n_offsets, n_joints, 4)
    if top_k is not None and top_k < flat.shape[1]:
        flat = flat[:, :top_k]
    return flat.astype(np.float64)
