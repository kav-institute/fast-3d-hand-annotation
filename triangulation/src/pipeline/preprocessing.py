"""
Detection preprocessing: undistortion, aggregation, candidate extraction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - OpenCV optional
    cv2 = None

from .geometry import huber_weight


def undistort_points(points: np.ndarray, K: np.ndarray, dist_coeff: np.ndarray) -> np.ndarray:
    """
    Undistort pixel coordinates using OpenCV if available. Points are expected as [..., 2].
    """
    if cv2 is None or dist_coeff.size == 0:
        return points
    reshaped = points.reshape(-1, 2)
    finite_mask = np.isfinite(reshaped[:, 0]) & np.isfinite(reshaped[:, 1])
    if not np.any(finite_mask):
        return points
    undistorted = reshaped.copy()
    undistorted_chunk = cv2.undistortPoints(
        reshaped[finite_mask].reshape(-1, 1, 2), K, dist_coeff, P=K
    )
    undistorted[finite_mask] = undistorted_chunk.reshape(-1, 2)
    return undistorted.reshape(points.shape)


def _cluster_candidates(
    pts: np.ndarray,
    scores: np.ndarray,
    radius_px: float,
) -> List[Dict[str, np.ndarray]]:
    if pts.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    assigned = np.zeros(len(pts), dtype=bool)
    clusters: List[Dict[str, np.ndarray]] = []
    for idx in order:
        if assigned[idx]:
            continue
        center = pts[idx]
        dists = np.linalg.norm(pts - center, axis=1)
        members = (dists <= radius_px) & ~assigned
        assigned[members] = True
        cluster_pts = pts[members]
        cluster_scores = scores[members].astype(np.float64)
        if cluster_pts.size == 0:
            continue
        if np.allclose(cluster_scores, cluster_scores[0]):
            weights = np.ones_like(cluster_scores)
        else:
            weights = cluster_scores
        w_sum = np.sum(weights)
        if w_sum < 1e-6:
            center = np.mean(cluster_pts, axis=0)
        else:
            center = (weights[:, None] * cluster_pts).sum(axis=0) / w_sum
        spread = np.median(np.linalg.norm(cluster_pts - center, axis=1))
        clusters.append(
            {
                "xy": center.astype(np.float64),
                "score": float(w_sum),
                "spread": float(spread),
            }
        )
    return clusters


def _kde_density(
    pts: np.ndarray,
    scores: np.ndarray,
    bandwidth_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if pts.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    diffs = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    denom = 2.0 * bandwidth_px * bandwidth_px
    weights = np.exp(-dist2 / denom) * scores[None, :]
    density = np.sum(weights, axis=1)
    spread = np.zeros_like(density)
    for i in range(len(pts)):
        w_sum = density[i]
        if w_sum < 1e-9:
            spread[i] = 0.0
        else:
            spread[i] = math.sqrt(float(np.sum(weights[i] * dist2[i]) / w_sum))
    return density, spread


def _kde_top_nms(
    pts: np.ndarray,
    scores: np.ndarray,
    max_candidates: Optional[int],
    bandwidth_px: float,
    nms_radius_px: float,
) -> List[Dict[str, np.ndarray]]:
    density, spread = _kde_density(pts, scores, bandwidth_px)
    if density.size == 0:
        return []
    order = np.argsort(density)[::-1]
    selected: List[Dict[str, np.ndarray]] = []
    selected_xy: List[np.ndarray] = []
    use_nms = nms_radius_px is not None and nms_radius_px > 0.0
    for idx in order:
        if max_candidates is not None and max_candidates > 0 and len(selected) >= max_candidates:
            break
        if use_nms and selected_xy:
            dists = np.linalg.norm(np.stack(selected_xy) - pts[idx], axis=1)
            if np.any(dists < nms_radius_px):
                continue
        selected_xy.append(pts[idx])
        selected.append(
            {
                "xy": pts[idx].astype(np.float64),
                "score": float(density[idx]),
                "spread": float(spread[idx]),
            }
        )
    return selected


def aggregate_candidates(
    candidates: np.ndarray,
    max_candidates: Optional[int] = 5,
    cluster_radius_px: Optional[float] = None,
    density_bandwidth_px: Optional[float] = None,
    density_nms_radius_px: Optional[float] = None,
    aggregate_mode: str = "robust",
) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, np.ndarray]]]]:
    """
    Convert [H, J, 4] hypotheses to a robust estimate and retain top-N raw candidates.

    Returns:
        xy: [J, 2] aggregated pixel locations (NaN if no valid hypothesis)
        sigma: [J] per-joint pixel sigma derived from hypothesis spread
        candidates_info: per-joint list of raw candidates with coordinates, score, and spread
    """
    hyp, joints, _ = candidates.shape
    xy_out = np.full((joints, 2), np.nan, dtype=np.float64)
    sigma_out = np.full((joints,), np.nan, dtype=np.float64)
    candidates_info: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(joints)]
    for j in range(joints):
        valid_mask = candidates[:, j, 3] > 0.5
        valid_idx = np.where(valid_mask)[0]
        if valid_idx.size == 0:
            continue
        pts = candidates[valid_idx, j, :2]
        scores = candidates[valid_idx, j, 2].astype(np.float64)
        if scores.ndim == 1:
            weights = scores.copy()
        else:
            weights = scores[:, 0].astype(np.float64)
        if np.allclose(weights, weights[0]):
            weights = np.ones_like(weights)
        if aggregate_mode == "top1":
            top_idx = int(np.argmax(scores))
            mean = pts[top_idx]
            spread = np.median(np.linalg.norm(pts - mean, axis=1))
        else:
            # Robust mean via iterative re-weighted least squares with Huber.
            mean = np.median(pts, axis=0)
            for _ in range(5):
                diffs = pts - mean
                residuals = np.linalg.norm(diffs, axis=1)
                huber = np.array([huber_weight(r, delta=10.0) for r in residuals])
                w = weights * huber
                w_sum = np.sum(w)
                if w_sum < 1e-6:
                    break
                mean = (w[:, None] * pts).sum(axis=0) / w_sum
            spread = np.median(np.linalg.norm(pts - mean, axis=1))
        xy_out[j] = mean
        sigma_out[j] = max(spread, 1.0)
        use_kde = density_bandwidth_px is not None and density_bandwidth_px > 0.0
        use_clusters = cluster_radius_px is not None and cluster_radius_px > 0.0
        if use_kde:
            nms_radius = density_nms_radius_px
            if nms_radius is None or nms_radius <= 0.0:
                nms_radius = float(density_bandwidth_px)
            cand_list = _kde_top_nms(
                pts,
                scores,
                max_candidates,
                float(density_bandwidth_px),
                float(nms_radius),
            )
            candidates_info[j].extend(cand_list)
        elif use_clusters:
            cand_list = _cluster_candidates(pts, scores, cluster_radius_px)
            cand_list.sort(key=lambda c: c["score"], reverse=True)
            if max_candidates is not None and max_candidates > 0:
                cand_list = cand_list[:max_candidates]
            candidates_info[j].extend(cand_list)
        else:
            order = np.argsort(scores)[::-1]
            if max_candidates is None or max_candidates <= 0:
                top_idx = order
            else:
                top_idx = order[:max_candidates]
            for idx in top_idx:
                original_idx = valid_idx[idx]
                cand_xy = candidates[original_idx, j, :2].astype(np.float64)
                candidates_info[j].append(
                    {
                        "xy": cand_xy,
                        "score": float(scores[idx]),
                        "spread": float(np.linalg.norm(cand_xy - mean)),
                    }
                )
    return xy_out, sigma_out, candidates_info
