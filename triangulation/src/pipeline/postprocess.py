"""
Minimal post-processing utilities for the clean triangulation pipeline.

Includes:
- temporal interpolation and smoothing with bone-length projection
- optional graph optimization stage (SciPy least squares)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

HAND_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (21, 22), (22, 23), (23, 24), (24, 25),
    (21, 26), (26, 27), (27, 28), (28, 29),
    (21, 30), (30, 31), (31, 32), (32, 33),
    (21, 34), (34, 35), (35, 36), (36, 37),
    (21, 38), (38, 39), (39, 40), (40, 41),
]


def interpolate_missing(positions: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fill missing joint positions by linear interpolation and edge extrapolation."""
    filled = positions.copy()
    updated_valid = valid_mask.copy()
    frames, joints, _ = positions.shape
    t = np.arange(frames)
    for j in range(joints):
        valid_frames = (
            updated_valid[:, j]
            & np.isfinite(filled[:, j, 0])
            & np.isfinite(filled[:, j, 1])
            & np.isfinite(filled[:, j, 2])
        )
        if not np.any(valid_frames):
            continue
        idx = np.where(valid_frames)[0]
        for axis in range(3):
            vals = filled[idx, j, axis]
            interp = np.interp(t, idx, vals)
            filled[:, j, axis] = interp
        updated_valid[:, j] = True
    return filled, updated_valid


def estimate_bone_lengths(positions: np.ndarray, valid_mask: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Estimate median bone length over frames where both endpoints are valid."""
    lengths: Dict[Tuple[int, int], List[float]] = {bone: [] for bone in HAND_BONES}
    frames = positions.shape[0]
    for t in range(frames):
        for a, b in HAND_BONES:
            if not valid_mask[t, a] or not valid_mask[t, b]:
                continue
            diff = positions[t, b] - positions[t, a]
            norm = float(np.linalg.norm(diff))
            if norm > 1e-6:
                lengths[(a, b)].append(norm)
    return {bone: (float(np.median(vals)) if vals else 0.0) for bone, vals in lengths.items()}


def enforce_bone_lengths(
    positions: np.ndarray,
    bone_lengths: Dict[Tuple[int, int], float],
    iterations: int = 5,
    lr: float = 0.25,
) -> np.ndarray:
    """Project joint pairs toward target bone lengths with iterative local updates."""
    refined = positions.copy()
    for _ in range(iterations):
        for (a, b), target_len in bone_lengths.items():
            if target_len <= 0.0:
                continue
            diff = refined[b] - refined[a]
            norm = float(np.linalg.norm(diff))
            if norm < 1e-6:
                continue
            correction = (norm - target_len) * (diff / norm)
            refined[a] += lr * correction
            refined[b] -= lr * correction
    return refined


def smooth_with_velocity_prior(
    track: np.ndarray,
    data_weight: np.ndarray,
    lambda_vel: float,
    iterations: int = 10,
) -> np.ndarray:
    """1D smoothing with data anchoring and first-order velocity prior."""
    x = track.copy()
    n = len(track)
    for _ in range(iterations):
        x_new = x.copy()
        for t in range(n):
            denom = data_weight[t]
            numer = data_weight[t] * track[t]
            if 0 < t < n - 1:
                numer += lambda_vel * (x[t - 1] + x[t + 1])
                denom += 2.0 * lambda_vel
            if denom > 1e-6:
                x_new[t] = numer / denom
        x = x_new
    return x


def sliding_window_refine(
    positions: np.ndarray,
    data_valid: np.ndarray,
    lambda_data: float = 1.0,
    lambda_vel: float = 0.2,
    window: int = 15,
    bone_lengths: Optional[Dict[Tuple[int, int], float]] = None,
) -> np.ndarray:
    """Temporal smoothing per joint/axis plus optional per-frame bone projection."""
    frames, joints, _ = positions.shape
    refined = positions.copy()
    data_weight = np.where(data_valid, lambda_data, lambda_data * 0.1).astype(np.float64)
    half = max(1, window // 2)
    for joint in range(joints):
        for axis in range(3):
            coord_track = refined[:, joint, axis]
            coord_weight = data_weight[:, joint]
            padded = coord_track.copy()
            for start in range(0, frames, half):
                end = min(start + window, frames)
                segment = padded[start:end]
                weights = coord_weight[start:end]
                padded[start:end] = smooth_with_velocity_prior(segment, weights, lambda_vel)
            refined[:, joint, axis] = padded
    if bone_lengths:
        for t in range(frames):
            refined[t] = enforce_bone_lengths(refined[t], bone_lengths)
    return refined


@dataclass
class GraphOptConfig:
    measurement_weight: float = 1.0
    smooth_weight: float = 0.3
    bone_weight: float = 1.0
    max_iterations: int = 20
    verbose: bool = False


class GraphPoseOptimizer:
    """Graph-style trajectory optimizer solved with SciPy least squares."""

    def __init__(self, joints: int, config: GraphOptConfig) -> None:
        self.joints = joints
        self.config = config
        self._eval_count = 0

    def _pack(self, positions: np.ndarray) -> np.ndarray:
        return positions.reshape(-1)

    def _unpack(self, vec: np.ndarray, frames: int) -> np.ndarray:
        return vec.reshape(frames, self.joints, 3)

    def _solve_window(
        self,
        positions: np.ndarray,
        validity: np.ndarray,
        sigmas: np.ndarray,
        bone_pairs: Sequence[Tuple[int, int]],
        bone_lengths: Dict[Tuple[int, int], float],
    ) -> np.ndarray:
        frames = positions.shape[0]
        mask_measure = validity.copy()
        measurement = positions.copy()
        sigma_valid_mask = np.isfinite(sigmas) & (sigmas > 0)
        finite_sigmas = sigmas[sigma_valid_mask]
        sigma_max = float(np.max(finite_sigmas)) if finite_sigmas.size > 0 else 1.0
        sigma_clipped = np.asarray(sigmas, dtype=np.float64).copy()
        sigma_clipped[~sigma_valid_mask] = sigma_max
        sigma_clipped[sigma_clipped <= 0.0] = 1e-3
        sigma_clipped = np.clip(sigma_clipped, 1e-3, sigma_max)
        smooth_mask = (
            np.ones((frames - 1, self.joints), dtype=bool)
            if frames > 1
            else np.zeros((0, self.joints), dtype=bool)
        )

        # Optimize only finite coordinates. A joint absent for the entire sequence
        # must remain missing, rather than making SciPy's initial vector non-finite.
        active = np.isfinite(positions)
        if not np.any(active):
            return positions.copy()

        def restore(vec: np.ndarray) -> np.ndarray:
            X = positions.copy()
            X[active] = vec
            return X

        def residual(vec: np.ndarray) -> np.ndarray:
            self._eval_count += 1
            X = restore(vec)
            res_list: List[np.ndarray] = []

            if self.config.measurement_weight > 0:
                meas_valid = (
                    mask_measure
                    & np.isfinite(measurement).all(axis=2)
                    & np.isfinite(X).all(axis=2)
                    & sigma_valid_mask
                )
                if np.any(meas_valid):
                    diff = (X - measurement)[meas_valid]
                    weights = (
                        self.config.measurement_weight
                        / np.maximum(sigma_clipped, 1e-3)
                    )[meas_valid][:, None]
                    res_list.append((diff * weights).reshape(-1))

            if self.config.smooth_weight > 0 and frames > 1:
                valid_prev = np.isfinite(X[:-1]).all(axis=2)
                valid_next = np.isfinite(X[1:]).all(axis=2)
                smooth_valid = smooth_mask & valid_prev & valid_next
                if np.any(smooth_valid):
                    delta = (X[1:] - X[:-1])[smooth_valid]
                    res_list.append((delta * np.sqrt(self.config.smooth_weight)).reshape(-1))

            if self.config.bone_weight > 0 and bone_pairs:
                weight = np.sqrt(self.config.bone_weight)
                bone_terms: List[np.ndarray] = []
                for a, b in bone_pairs:
                    Xa = X[:, a, :]
                    Xb = X[:, b, :]
                    mask = np.isfinite(Xa).all(axis=1) & np.isfinite(Xb).all(axis=1)
                    if not np.any(mask):
                        continue
                    target = bone_lengths.get((a, b)) or bone_lengths.get((b, a)) or 0.0
                    if target <= 0.0:
                        continue
                    diff = np.linalg.norm(Xa[mask] - Xb[mask], axis=1) - target
                    bone_terms.append(weight * diff)
                if bone_terms:
                    res_list.append(np.concatenate(bone_terms))

            if not res_list:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate(res_list)

        x0 = positions[active]
        if residual(x0).size == 0:
            return positions.copy()
        result = least_squares(
            residual,
            x0,
            max_nfev=self.config.max_iterations,
            method="trf",
            verbose=2 if self.config.verbose else 0,
        )
        return restore(result.x)

    def optimize(
        self,
        positions: np.ndarray,
        validity: np.ndarray,
        sigmas: np.ndarray,
        bone_pairs: Sequence[Tuple[int, int]],
        bone_lengths: Dict[Tuple[int, int], float],
        window: Optional[int] = None,
        overlap: int = 0,
    ) -> np.ndarray:
        frames = positions.shape[0]
        if window is None or window <= 0 or window >= frames:
            return self._solve_window(positions, validity, sigmas, bone_pairs, bone_lengths)
        overlap = max(0, min(overlap, window - 1))
        refined = np.copy(positions)
        start = 0
        while start < frames:
            end = min(start + window, frames)
            refined_chunk = self._solve_window(
                positions[start:end],
                validity[start:end],
                sigmas[start:end],
                bone_pairs,
                bone_lengths,
            )
            refined[start:end] = refined_chunk
            if end == frames:
                break
            start = end - overlap
        return refined


def run_post_process(
    positions_raw: np.ndarray,
    validity: np.ndarray,
    residual_sigmas: np.ndarray,
    manual_annotations: Optional[np.ndarray] = None,
    lambda_vel: float = 0.2,
    graph_window: int = 10,
    graph_overlap: int = 5,
    graph_config: Optional[GraphOptConfig] = None,
) -> Dict[str, np.ndarray]:
    """Run clean post stage: interpolate -> refine -> optional graph optimize."""
    positions_filled, filled_valid = interpolate_missing(positions_raw, validity)
    bone_lengths = estimate_bone_lengths(positions_filled, filled_valid)
    positions_refined = sliding_window_refine(
        positions_filled,
        filled_valid,
        lambda_data=1.0,
        lambda_vel=lambda_vel,
        bone_lengths=bone_lengths,
    )
    if manual_annotations is None or np.shape(manual_annotations) != np.shape(validity):
        manual_annotations = np.zeros_like(validity, dtype=bool)
    else:
        manual_annotations = np.asarray(manual_annotations, dtype=bool)

    results: Dict[str, np.ndarray] = {
        "positions_raw": np.asarray(positions_raw, dtype=np.float64),
        "positions_interpolated": np.asarray(positions_filled, dtype=np.float64),
        "positions_refined": np.asarray(positions_refined, dtype=np.float64),
        "validity": np.asarray(filled_valid, dtype=bool),
        "manual_annotations": manual_annotations,
        "residual_sigmas": np.asarray(residual_sigmas, dtype=np.float64),
    }
    if graph_config is not None:
        optimizer = GraphPoseOptimizer(positions_raw.shape[1], graph_config)
        positions_graph = optimizer.optimize(
            positions_refined,
            filled_valid,
            residual_sigmas,
            HAND_BONES,
            bone_lengths,
            window=graph_window if graph_window > 0 else None,
            overlap=graph_overlap,
        )
        results["positions_graph"] = np.asarray(positions_graph, dtype=np.float64)
    return results

