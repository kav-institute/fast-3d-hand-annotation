"""
Geometry helpers for multi-view triangulation and projection.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def skew(vec: np.ndarray) -> np.ndarray:
    """Return the 3x3 cross-product (skew symmetric) matrix of a 3-vector."""
    x, y, z = vec
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def homogenize(points2d: np.ndarray) -> np.ndarray:
    """Append 1 to each 2D point."""
    ones = np.ones((*points2d.shape[:-1], 1), dtype=points2d.dtype)
    return np.concatenate([points2d, ones], axis=-1)


def normalize_homogeneous(vec: np.ndarray) -> np.ndarray:
    """Normalize a homogeneous vector so the last coordinate is 1."""
    if abs(vec[-1]) < 1e-12:
        return vec.copy()
    return vec / vec[-1]


def project_point(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D homogeneous point X with projection matrix P."""
    x = P @ X
    z = float(x[2])
    if not np.isfinite(z) or abs(z) < 1e-9:
        return np.array([np.nan, np.nan], dtype=np.float64)
    xy = x[:2] / z
    if not np.all(np.isfinite(xy)):
        return np.array([np.nan, np.nan], dtype=np.float64)
    return xy.astype(np.float64)


def triangulate_linear(
    projections: Sequence[np.ndarray],
    points: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Solve for homogeneous 3D point using linear least squares (DLT).
    projections: list of 3x4 matrices.
    points: list of (u, v) observations.
    """
    assert len(projections) == len(points)
    A_rows: List[np.ndarray] = []
    for P, (u, v) in zip(projections, points):
        A_rows.append(u * P[2] - P[0])
        A_rows.append(v * P[2] - P[1])
    A = np.stack(A_rows, axis=0)
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    return normalize_homogeneous(X)


def weighted_triangulate(
    projections: Sequence[np.ndarray],
    points: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """Weighted DLT refinement with per-view scalar weights."""
    A_rows: List[np.ndarray] = []
    for P, (u, v), w in zip(projections, points, weights):
        w = max(w, 1e-6) ** 0.5
        A_rows.append(w * (u * P[2] - P[0]))
        A_rows.append(w * (v * P[2] - P[1]))
    A = np.stack(A_rows, axis=0)
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    return normalize_homogeneous(X)


def compute_fundamental_from_projections(P_a: np.ndarray, P_b: np.ndarray) -> np.ndarray:
    """
    Build the fundamental matrix F_ab from projection matrices P_a, P_b using the
    classic relationship F = [e_b]_x P_b P_a^+, where e_b is the epipole of camera a in view b.
    """
    _, _, vh = np.linalg.svd(P_a)
    C_a = normalize_homogeneous(vh[-1])
    e_b = P_b @ C_a
    e_b = e_b[:3] / e_b[2]
    F = skew(e_b) @ P_b @ np.linalg.pinv(P_a)
    return F / np.linalg.norm(F)


def epipolar_distance(x_a: np.ndarray, x_b: np.ndarray, F: np.ndarray) -> float:
    """
    Normalized symmetric epipolar distance between two pixels given F.
    Formula matches Hartley & Zisserman (2003).
    """
    x_a_h = homogenize(x_a)
    x_b_h = homogenize(x_b)
    Fx_a = F @ x_a_h
    F_tx_b = F.T @ x_b_h
    numerator = float(x_b_h.T @ F @ x_a_h)
    denom = math.sqrt(Fx_a[0] ** 2 + Fx_a[1] ** 2) + math.sqrt(F_tx_b[0] ** 2 + F_tx_b[1] ** 2)
    if denom < 1e-9:
        return float("inf")
    return abs(numerator) / denom


def huber_weight(residual: float, delta: float) -> float:
    """Compute scalar weight for Huber loss."""
    abs_r = abs(residual)
    if abs_r <= delta:
        return 1.0
    return delta / abs_r
