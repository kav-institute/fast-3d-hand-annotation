#!/usr/bin/env python3
"""
Evaluate 2D detection candidate sets against projected 3D GT.

This script projects 3D GT joints into each calibrated camera and compares them
to the raw 2D detector samples (candidate hypotheses) stored in
``detections*_camera_*_2d.npy``.

Supported 3D GT inputs:
- Triangulation result NPZ (e.g. ``triangulation_pipeline_outputs.npz``) with a key
  such as ``positions_graph`` / ``positions_refined`` / ``positions_raw``.
- JSON GT with frame keys and optional ``world_coord`` + ``joint_valid`` entries.

By default, the comparison is performed in the same undistorted pixel space used
by triangulation (detections are undistorted using the camera intrinsics).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.dataloading import CameraModel, load_camera_parameters, load_detection_tensor
from pipeline.geometry import project_point
from pipeline.preprocessing import undistort_points


def _sort_frame_keys(keys: Sequence[str]) -> List[str]:
    def key_sort_value(k: str) -> Tuple[int, object]:
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    return sorted(keys, key=key_sort_value)


def _slice_frames(
    positions: np.ndarray,
    validity: np.ndarray,
    frame_start: int,
    frame_count: Optional[int],
    frame_stride: int,
    frame_keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    if frame_stride < 1:
        frame_stride = 1
    total = positions.shape[0]
    if frame_start < 0 or frame_start >= total:
        raise ValueError(f"frame_start={frame_start} outside [0, {total})")
    end = total if frame_count is None else min(total, frame_start + frame_count * frame_stride)
    sl = slice(frame_start, end, frame_stride)
    sliced_pos = positions[sl]
    sliced_valid = validity[sl]
    sliced_keys = frame_keys[sl] if frame_keys is not None else None
    return sliced_pos, sliced_valid, sliced_keys


def _load_gt_npz(
    path: Path,
    key: str,
    validity_override_path: Optional[Path],
    validity_override_key: str,
    auto_label_stage_validity: bool,
    frame_start: int,
    frame_count: Optional[int],
    frame_stride: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")
    positions = np.asarray(data[key], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"Expected [F, J, 3] for '{key}' in {path}, got {positions.shape}")
    validity = data.get("validity")

    # Post-stage pipeline outputs store a filled/interpolated validity mask (often all True).
    # For evaluating projected 3D against 2D detections, the original label-stage validity is
    # usually the correct mask. Allow an explicit override and also auto-detect a sibling file.
    override_path = validity_override_path
    if override_path is None and auto_label_stage_validity and key in {
        "positions_interpolated",
        "positions_refined",
        "positions_graph",
        "fk_positions",
        "fk_positions_smoothed",
    }:
        sibling = path.parent / "triangulation_label_stage.npz"
        if sibling.exists():
            override_path = sibling
            print(
                f"[info] Using validity mask from sibling label-stage file: "
                f"{override_path} (key='{validity_override_key}')"
            )

    if override_path is not None:
        with np.load(override_path, allow_pickle=True) as vdata:
            if validity_override_key not in vdata:
                raise KeyError(
                    f"Validity key '{validity_override_key}' not found in {override_path}"
                )
            validity = np.asarray(vdata[validity_override_key], dtype=bool)

    if validity is None or np.shape(validity)[:2] != positions.shape[:2]:
        validity_mask = np.isfinite(positions).all(axis=2)
    else:
        validity_mask = np.asarray(validity, dtype=bool)
    return _slice_frames(positions, validity_mask, frame_start, frame_count, frame_stride)


def _load_gt_json(
    path: Path,
    frame_start: int,
    frame_count: Optional[int],
    frame_stride: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object of frames in {path}")
    keys = _sort_frame_keys(list(data.keys()))
    if not keys:
        raise ValueError(f"No frames found in {path}")

    frames: List[np.ndarray] = []
    valid_frames: List[np.ndarray] = []
    for key in keys:
        entry = data[key]
        if isinstance(entry, dict) and "world_coord" in entry:
            frame = np.asarray(entry["world_coord"], dtype=np.float64)
            joint_valid = entry.get("joint_valid")
        else:
            frame = np.asarray(entry, dtype=np.float64)
            joint_valid = None
        if frame.ndim != 2 or frame.shape[1] != 3:
            raise ValueError(f"Frame {key} has invalid shape {frame.shape} in {path}")
        if isinstance(entry, dict) and entry.get("hand_type_valid") is False:
            valid = np.zeros(frame.shape[0], dtype=bool)
        elif joint_valid is None:
            valid = np.isfinite(frame).all(axis=1)
        else:
            valid_arr = np.asarray(joint_valid, dtype=np.float64).reshape(-1)
            if valid_arr.shape[0] != frame.shape[0]:
                raise ValueError(
                    f"joint_valid length {valid_arr.shape[0]} does not match joints "
                    f"{frame.shape[0]} for frame {key}"
                )
            valid = valid_arr > 0.5
        frames.append(frame)
        valid_frames.append(valid)
    positions = np.stack(frames, axis=0)
    validity = np.stack(valid_frames, axis=0)
    return _slice_frames(positions, validity, frame_start, frame_count, frame_stride, frame_keys=keys)  # type: ignore[return-value]


def load_gt_3d(
    path: Path,
    key: str,
    validity_override_path: Optional[Path],
    validity_override_key: str,
    auto_label_stage_validity: bool,
    frame_start: int,
    frame_count: Optional[int],
    frame_stride: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_gt_npz(
            path,
            key,
            validity_override_path,
            validity_override_key,
            auto_label_stage_validity,
            frame_start,
            frame_count,
            frame_stride,
        )
    if suffix == ".json":
        return _load_gt_json(path, frame_start, frame_count, frame_stride)
    raise ValueError(f"Unsupported GT file type '{suffix}' for {path}; expected .npz or .json")


def _load_param_file(scene_root: Path) -> Path:
    matches = list(scene_root.glob("camera_parameter*.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one camera parameter file in {scene_root}, found: {matches}"
        )
    return matches[0]


def _load_image_resolution(param_path: Path) -> Optional[Tuple[float, float]]:
    """Read [width, height] from camera-parameter header if available."""
    with param_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    resolution = data.get("header", {}).get("resolution")
    if not isinstance(resolution, list) or len(resolution) < 2:
        return None
    width = float(resolution[0])
    height = float(resolution[1])
    if width <= 0.0 or height <= 0.0:
        return None
    return width, height


def _detection_glob_and_split(detection_source: str) -> Tuple[str, int]:
    if detection_source == "default":
        return "detections_camera_*_2d.npy", 2
    if detection_source == "sapiens":
        return "detections_sp_camera_*_2d.npy", 3
    if detection_source == "mediapipe":
        return "detections_mp_camera_*_2d.npy", 3
    if detection_source == "interwild":
        return "detections_iw_camera_*_2d.npy", 3
    raise ValueError(f"Unsupported detection_source '{detection_source}'")


def _safe_project(cam: CameraModel, X: np.ndarray) -> np.ndarray:
    """Project a 3D point and avoid divide-by-zero warnings for invalid depth."""
    X_h = np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    x = cam.P @ X_h
    z = float(x[2])
    if not np.isfinite(z) or abs(z) < 1e-9:
        return np.array([np.nan, np.nan], dtype=np.float64)
    # Keep project_point in the path for consistency with the rest of the codebase.
    return project_point(cam.P, X_h)


def load_detection_tensors(
    scene_root: Path,
    detection_source: str,
    top_k_candidates: Optional[int],
    camera_filter: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    pattern, split_idx = _detection_glob_and_split(detection_source)
    detections: Dict[str, np.ndarray] = {}
    camera_filter_set = set(camera_filter) if camera_filter else None
    for det_path in sorted(scene_root.glob(pattern)):
        parts = det_path.stem.split("_")
        if split_idx >= len(parts):
            continue
        cam_id = parts[split_idx]
        if camera_filter_set is not None and cam_id not in camera_filter_set:
            continue
        detections[cam_id] = load_detection_tensor(det_path, top_k=top_k_candidates)
    if not detections:
        raise RuntimeError(f"No detection tensors found for source '{detection_source}' in {scene_root}")
    return detections


@dataclass
class CandidateEvalStats:
    total_queries: int = 0
    with_candidates: int = 0
    best_errors: List[float] = field(default_factory=list)
    first_errors: List[float] = field(default_factory=list)
    topscore_errors: List[float] = field(default_factory=list)
    hand_total_queries: int = 0
    best_hand_errors: List[float] = field(default_factory=list)
    first_hand_errors: List[float] = field(default_factory=list)
    topscore_hand_errors: List[float] = field(default_factory=list)
    candidate_counts: List[int] = field(default_factory=list)
    frame_count: int = 0
    auto_swapped_frames: int = 0

    def add_missing(self, hand_scale: Optional[float] = None) -> None:
        self.total_queries += 1
        if hand_scale is not None and np.isfinite(hand_scale) and hand_scale > 0.0:
            self.hand_total_queries += 1

    def add(
        self,
        best: float,
        first: float,
        topscore: float,
        candidate_count: int,
        hand_scale: Optional[float] = None,
    ) -> None:
        self.total_queries += 1
        self.with_candidates += 1
        self.best_errors.append(float(best))
        self.first_errors.append(float(first))
        self.topscore_errors.append(float(topscore))
        self.candidate_counts.append(int(candidate_count))
        if hand_scale is not None and np.isfinite(hand_scale) and hand_scale > 0.0:
            self.hand_total_queries += 1
            self.best_hand_errors.append(100.0 * float(best) / float(hand_scale))
            self.first_hand_errors.append(100.0 * float(first) / float(hand_scale))
            self.topscore_hand_errors.append(100.0 * float(topscore) / float(hand_scale))

    def merge(self, other: "CandidateEvalStats") -> None:
        self.total_queries += other.total_queries
        self.with_candidates += other.with_candidates
        self.best_errors.extend(other.best_errors)
        self.first_errors.extend(other.first_errors)
        self.topscore_errors.extend(other.topscore_errors)
        self.hand_total_queries += other.hand_total_queries
        self.best_hand_errors.extend(other.best_hand_errors)
        self.first_hand_errors.extend(other.first_hand_errors)
        self.topscore_hand_errors.extend(other.topscore_hand_errors)
        self.candidate_counts.extend(other.candidate_counts)
        self.frame_count += other.frame_count
        self.auto_swapped_frames += other.auto_swapped_frames


def _hand_joint_groups(joint_count: int) -> List[range]:
    if joint_count == 42:
        return [range(0, 21), range(21, 42)]
    return [range(0, joint_count)]


def _compute_hand_scales(
    projected_gt: np.ndarray,
    projected_valid: np.ndarray,
    min_joints: int,
    min_scale_px: float,
) -> np.ndarray:
    """Return per-joint GT hand bbox scale in pixels, using longest bbox side."""
    joint_count = projected_gt.shape[0]
    scales = np.full(joint_count, np.nan, dtype=np.float64)
    for group in _hand_joint_groups(joint_count):
        group_idx = np.asarray(list(group), dtype=int)
        valid_idx = group_idx[projected_valid[group_idx]]
        if valid_idx.size < min_joints:
            continue
        pts = projected_gt[valid_idx]
        if pts.size == 0 or not np.all(np.isfinite(pts)):
            continue
        width = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        height = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        scale = max(width, height)
        if np.isfinite(scale) and scale >= min_scale_px:
            scales[group_idx] = scale
    return scales


def _swap_joint_index(joint_idx: int, joint_count: int) -> Optional[int]:
    # Current hand layout is expected to be 42 joints = [0..20] right, [21..41] left.
    if joint_count != 42:
        return None
    if 0 <= joint_idx < 21:
        return joint_idx + 21
    if 21 <= joint_idx < 42:
        return joint_idx - 21
    return None


def _metric_summary(errors: Sequence[float], total_queries: int, thresholds: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64)
    out: Dict[str, float] = {}
    if arr.size == 0:
        out["mean"] = float("nan")
        out["median"] = float("nan")
        for t in thresholds:
            out[f"pck@{t:g}"] = 0.0 if total_queries > 0 else float("nan")
        return out
    out["mean"] = float(np.mean(arr))
    out["median"] = float(np.median(arr))
    for t in thresholds:
        out[f"pck@{t:g}"] = float(np.sum(arr <= t) / total_queries) if total_queries > 0 else float("nan")
    return out


def _diag_metric_summary(
    errors: Sequence[float],
    total_queries: int,
    image_diagonal_px: float,
    thresholds_percent: Sequence[float],
) -> Dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64)
    out: Dict[str, float] = {}
    if arr.size == 0:
        out["NME_diag_mean"] = float("nan")
        out["NME_diag_median"] = float("nan")
        for t in thresholds_percent:
            out[f"PCK@{t:g}%diag"] = 0.0 if total_queries > 0 else float("nan")
        return out

    normalized_percent = 100.0 * arr / image_diagonal_px
    out["NME_diag_mean"] = float(np.mean(normalized_percent))
    out["NME_diag_median"] = float(np.median(normalized_percent))
    for t in thresholds_percent:
        out[f"PCK@{t:g}%diag"] = (
            float(np.sum(normalized_percent <= t) / total_queries)
            if total_queries > 0
            else float("nan")
        )
    return out


def _hand_metric_summary(
    errors_percent: Sequence[float],
    total_queries: int,
    thresholds_percent: Sequence[float],
) -> Dict[str, float]:
    arr = np.asarray(errors_percent, dtype=np.float64)
    out: Dict[str, float] = {}
    if arr.size == 0:
        out["NME_hand_mean"] = float("nan")
        out["NME_hand_median"] = float("nan")
        for t in thresholds_percent:
            out[f"PCK@{t:g}%hand"] = 0.0 if total_queries > 0 else float("nan")
        return out

    out["NME_hand_mean"] = float(np.mean(arr))
    out["NME_hand_median"] = float(np.median(arr))
    for t in thresholds_percent:
        out[f"PCK@{t:g}%hand"] = (
            float(np.sum(arr <= t) / total_queries)
            if total_queries > 0
            else float("nan")
        )
    return out


def _format_float(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.3f}"


def _print_stats(
    name: str,
    stats: CandidateEvalStats,
    thresholds: Sequence[float],
    image_diagonal_px: Optional[float] = None,
    diag_thresholds_percent: Sequence[float] = (),
    hand_thresholds_percent: Sequence[float] = (),
) -> None:
    coverage = (stats.with_candidates / stats.total_queries) if stats.total_queries > 0 else float("nan")
    cand_counts = np.asarray(stats.candidate_counts, dtype=np.float64)
    cand_mean = float(np.mean(cand_counts)) if cand_counts.size else float("nan")
    cand_median = float(np.median(cand_counts)) if cand_counts.size else float("nan")

    best = _metric_summary(stats.best_errors, stats.total_queries, thresholds)
    first = _metric_summary(stats.first_errors, stats.total_queries, thresholds)
    top = _metric_summary(stats.topscore_errors, stats.total_queries, thresholds)

    print(f"\n{name}")
    print(f"  queries={stats.total_queries} with_candidates={stats.with_candidates} coverage={_format_float(coverage)}")
    print(f"  candidate_count(mean/median)={_format_float(cand_mean)}/{_format_float(cand_median)}")
    if hand_thresholds_percent:
        print(f"  hand_scale_queries={stats.hand_total_queries}/{stats.total_queries}")
    if stats.frame_count > 0:
        print(
            "  auto_swap_frames="
            f"{stats.auto_swapped_frames}/{stats.frame_count}"
            f" ({_format_float(stats.auto_swapped_frames / stats.frame_count)})"
        )

    def print_line(label: str, metrics: Dict[str, float]) -> None:
        parts = [
            f"{label}: mean={_format_float(metrics['mean'])}",
            f"median={_format_float(metrics['median'])}",
        ]
        for t in thresholds:
            parts.append(f"pck@{t:g}={_format_float(metrics[f'pck@{t:g}'])}")
        print("  " + " | ".join(parts))

    print_line("oracle(min)", best)
    print_line("first_valid", first)
    print_line("top_score", top)

    if image_diagonal_px is not None and image_diagonal_px > 0.0:
        best_diag = _diag_metric_summary(
            stats.best_errors, stats.total_queries, image_diagonal_px, diag_thresholds_percent
        )
        first_diag = _diag_metric_summary(
            stats.first_errors, stats.total_queries, image_diagonal_px, diag_thresholds_percent
        )
        top_diag = _diag_metric_summary(
            stats.topscore_errors, stats.total_queries, image_diagonal_px, diag_thresholds_percent
        )

        def print_diag_line(label: str, metrics: Dict[str, float]) -> None:
            parts = [
                f"{label}: NME_diag_mean={_format_float(metrics['NME_diag_mean'])}%",
                f"NME_diag_median={_format_float(metrics['NME_diag_median'])}%",
            ]
            for t in diag_thresholds_percent:
                parts.append(f"PCK@{t:g}%diag={_format_float(metrics[f'PCK@{t:g}%diag'])}")
            print("  " + " | ".join(parts))

        print_diag_line("oracle(min) normalized", best_diag)
        print_diag_line("first_valid normalized", first_diag)
        print_diag_line("top_score normalized", top_diag)

    if hand_thresholds_percent:
        best_hand = _hand_metric_summary(
            stats.best_hand_errors, stats.hand_total_queries, hand_thresholds_percent
        )
        first_hand = _hand_metric_summary(
            stats.first_hand_errors, stats.hand_total_queries, hand_thresholds_percent
        )
        top_hand = _hand_metric_summary(
            stats.topscore_hand_errors, stats.hand_total_queries, hand_thresholds_percent
        )

        def print_hand_line(label: str, metrics: Dict[str, float]) -> None:
            parts = [
                f"{label}: NME_hand_mean={_format_float(metrics['NME_hand_mean'])}%",
                f"NME_hand_median={_format_float(metrics['NME_hand_median'])}%",
            ]
            for t in hand_thresholds_percent:
                parts.append(f"PCK@{t:g}%hand={_format_float(metrics[f'PCK@{t:g}%hand'])}")
            print("  " + " | ".join(parts))

        print_hand_line("oracle(min) hand-normalized", best_hand)
        print_hand_line("first_valid hand-normalized", first_hand)
        print_hand_line("top_score hand-normalized", top_hand)


def evaluate_camera(
    cam_id: str,
    cam: CameraModel,
    det_tensor: np.ndarray,
    gt_positions: np.ndarray,
    gt_validity: np.ndarray,
    det_valid_threshold: float,
    compare_undistorted: bool,
    hand_scale_min_joints: int = 5,
    hand_scale_min_px: float = 1.0,
    hand_swap_mode: str = "none",
    exclude_frames: Optional[Sequence[int]] = None,
    exclude_joints: Optional[Sequence[int]] = None,
) -> CandidateEvalStats:
    frames_gt, joints_gt, _ = gt_positions.shape
    frames_det, _, joints_det, _ = det_tensor.shape
    if joints_det != joints_gt:
        raise ValueError(
            f"Joint count mismatch for camera {cam_id}: detections={joints_det}, gt={joints_gt}"
        )
    frames = min(frames_gt, frames_det)
    if frames <= 0:
        return CandidateEvalStats()

    stats = CandidateEvalStats()
    exclude_set = set(int(x) for x in (exclude_frames or []))
    exclude_joint_set = {
        int(x) for x in (exclude_joints or [])
        if 0 <= int(x) < joints_gt
    }
    for f in range(frames):
        if f in exclude_set:
            continue
        det_frame = np.asarray(det_tensor[f], dtype=np.float64)  # [H, J, 4]
        cand_xy = det_frame[:, :, :2]
        if compare_undistorted:
            cand_xy = undistort_points(cand_xy, cam.K, cam.dist_coeff)
        cand_scores = det_frame[:, :, 2]
        cand_valid = det_frame[:, :, 3] > det_valid_threshold

        projected_gt = np.full((joints_gt, 2), np.nan, dtype=np.float64)
        projected_valid = np.zeros(joints_gt, dtype=bool)
        for j in range(joints_gt):
            if not bool(gt_validity[f, j]):
                continue
            X = gt_positions[f, j]
            if not np.all(np.isfinite(X)):
                continue
            proj = _safe_project(cam, X)
            if not np.all(np.isfinite(proj)):
                continue
            projected_gt[j] = proj
            projected_valid[j] = True

        if exclude_joint_set:
            projected_valid[list(exclude_joint_set)] = False

        hand_scales = _compute_hand_scales(
            projected_gt,
            projected_valid,
            min_joints=max(1, int(hand_scale_min_joints)),
            min_scale_px=max(0.0, float(hand_scale_min_px)),
        )

        stats.frame_count += 1
        use_swap_for_frame = (hand_swap_mode == "global")
        if hand_swap_mode == "auto" and joints_gt == 42:
            direct_errors: List[float] = []
            swapped_errors: List[float] = []
            for j in range(joints_gt):
                if j in exclude_joint_set:
                    continue
                if not projected_valid[j]:
                    continue
                proj = projected_gt[j]

                valid_idx_direct = np.where(cand_valid[:, j])[0]
                if valid_idx_direct.size > 0:
                    d = np.linalg.norm(cand_xy[valid_idx_direct, j, :] - proj[None, :], axis=1)
                    if np.any(np.isfinite(d)):
                        direct_errors.append(float(np.nanmin(d)))

                j_swap = _swap_joint_index(j, joints_gt)
                if j_swap is None:
                    continue
                valid_idx_swap = np.where(cand_valid[:, j_swap])[0]
                if valid_idx_swap.size > 0:
                    d = np.linalg.norm(cand_xy[valid_idx_swap, j_swap, :] - proj[None, :], axis=1)
                    if np.any(np.isfinite(d)):
                        swapped_errors.append(float(np.nanmin(d)))

            if direct_errors and swapped_errors:
                direct_mean = float(np.mean(direct_errors))
                swapped_mean = float(np.mean(swapped_errors))
                use_swap_for_frame = swapped_mean < direct_mean
            else:
                use_swap_for_frame = False
        if use_swap_for_frame and hand_swap_mode == "auto":
            stats.auto_swapped_frames += 1

        for j in range(joints_gt):
            if j in exclude_joint_set:
                continue
            if not projected_valid[j]:
                continue
            proj = projected_gt[j]
            hand_scale = float(hand_scales[j]) if np.isfinite(hand_scales[j]) else None

            j_eval = j
            if use_swap_for_frame:
                j_swap = _swap_joint_index(j, joints_gt)
                if j_swap is not None:
                    j_eval = j_swap

            valid_idx = np.where(cand_valid[:, j_eval])[0]
            if valid_idx.size == 0:
                stats.add_missing(hand_scale=hand_scale)
                continue

            pts = cand_xy[valid_idx, j_eval, :]
            dists = np.linalg.norm(pts - proj[None, :], axis=1)
            if not np.any(np.isfinite(dists)):
                stats.add_missing(hand_scale=hand_scale)
                continue
            finite_mask = np.isfinite(dists)
            dists = dists[finite_mask]
            valid_idx_f = valid_idx[finite_mask]
            if dists.size == 0:
                stats.add_missing(hand_scale=hand_scale)
                continue

            best_err = float(np.min(dists))
            first_err = float(dists[0])
            score_vals = np.asarray(cand_scores[valid_idx_f, j_eval], dtype=np.float64)
            if score_vals.size == 0 or not np.any(np.isfinite(score_vals)):
                top_err = best_err
            else:
                top_local = int(np.nanargmax(score_vals))
                top_err = float(dists[top_local])
            stats.add(
                best=best_err,
                first=first_err,
                topscore=top_err,
                candidate_count=int(dists.size),
                hand_scale=hand_scale,
            )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True, help="Scenario root containing camera params and detection tensors.")
    parser.add_argument(
        "--gt-3d",
        type=Path,
        required=True,
        help="3D GT source (.npz triangulation output or .json GT file).",
    )
    parser.add_argument(
        "--gt-key",
        type=str,
        default="positions_graph",
        help="Key to read from GT npz (ignored for JSON).",
    )
    parser.add_argument(
        "--gt-validity-npz",
        type=Path,
        default=None,
        help=(
            "Optional NPZ providing the validity mask for GT npz evaluation "
            "(e.g. outputs_*/triangulation_label_stage.npz)."
        ),
    )
    parser.add_argument(
        "--gt-validity-key",
        type=str,
        default="validity",
        help="Key to read validity mask from --gt-validity-npz (default: validity).",
    )
    parser.add_argument(
        "--no-auto-label-stage-validity",
        action="store_true",
        help=(
            "Disable automatic fallback to sibling triangulation_label_stage.npz validity "
            "when evaluating post-stage keys like positions_graph."
        ),
    )
    parser.add_argument(
        "--camera-params",
        type=Path,
        default=None,
        help="Camera parameter JSON (default: auto-detect camera_parameter*.json under scene-root).",
    )
    parser.add_argument(
        "--image-resolution",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help=(
            "Image resolution used for diagonal-normalized metrics. "
            "Default: read header.resolution from the camera parameter JSON."
        ),
    )
    parser.add_argument(
        "--detection-source",
        choices=["default", "sapiens", "mediapipe", "interwild"],
        default="default",
        help="Which detection tensor naming scheme to load.",
    )
    parser.add_argument(
        "--camera",
        nargs="+",
        default=None,
        help="Optional camera id subset.",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=0,
        help="Limit candidate count per frame before evaluation (0 = all).",
    )
    parser.add_argument(
        "--det-valid-threshold",
        type=float,
        default=0.5,
        help="Detection valid flag threshold (candidate[...,3] > threshold).",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Start frame index (for NPZ) or sorted JSON-frame index/key position.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help="Optional number of frames to evaluate after start/stride.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Evaluate every n-th frame (applied to GT; detections are aligned by order).",
    )
    parser.add_argument(
        "--compare-space",
        choices=["undistorted"],
        default="undistorted",
        help="Coordinate space used for comparison (currently only undistorted, matching triangulation).",
    )
    parser.add_argument(
        "--hand-swap",
        choices=["none", "global", "auto"],
        default="none",
        help=(
            "How to handle possible left/right hand slot swaps for 42-joint layouts: "
            "'global' swaps all joints (0..20)<->(21..41), 'auto' picks direct vs swapped "
            "per camera-frame based on lower mean oracle error."
        ),
    )
    parser.add_argument(
        "--pck-thresholds",
        nargs="+",
        type=float,
        default=[4.0, 8.0, 16.0, 32.0, 64.0], 
        help="2D pixel thresholds for PCK-like candidate coverage metrics.",
    )
    parser.add_argument(
        "--diag-pck-thresholds-percent",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 5.0],
        help="PCK thresholds as percentages of the image diagonal for normalized reporting.",
    )
    parser.add_argument(
        "--hand-pck-thresholds-percent",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0],
        help="PCK thresholds as percentages of the projected GT hand bbox longest side.",
    )
    parser.add_argument(
        "--hand-scale-min-joints",
        type=int,
        default=5,
        help="Minimum valid projected GT joints required to define a hand bbox scale.",
    )
    parser.add_argument(
        "--hand-scale-min-px",
        type=float,
        default=1.0,
        help="Minimum GT hand bbox longest side in pixels required for hand-normalized metrics.",
    )
    parser.add_argument(
        "--exclude-joints",
        nargs="*",
        type=int,
        default=[0, 21],
        help=(
            "Joint indices to exclude from 2D metrics and hand-scale bboxes "
            "(default: 0 21, the wrist joints in the 42-joint hand layout)."
        ),
    )
    parser.add_argument(
        "--exclude-frames",
        nargs="*",
        type=int,
        default=[],
        help="Frame indices to exclude from evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_root = args.scene_root
    if args.camera_params is None:
        args.camera_params = _load_param_file(scene_root)

    if args.image_resolution is not None:
        image_resolution = (float(args.image_resolution[0]), float(args.image_resolution[1]))
    else:
        image_resolution = _load_image_resolution(args.camera_params)
    image_diagonal_px: Optional[float] = None
    if image_resolution is not None:
        width, height = image_resolution
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Invalid image resolution: width={width}, height={height}")
        image_diagonal_px = float(np.hypot(width, height))

    gt_positions, gt_validity, frame_keys = load_gt_3d(
        args.gt_3d,
        key=args.gt_key,
        validity_override_path=args.gt_validity_npz,
        validity_override_key=args.gt_validity_key,
        auto_label_stage_validity=not args.no_auto_label_stage_validity,
        frame_start=args.frame_start,
        frame_count=args.frame_count,
        frame_stride=args.frame_stride,
    )
    if gt_positions.shape[:2] != gt_validity.shape:
        raise ValueError(
            f"GT validity shape mismatch: positions={gt_positions.shape}, validity={gt_validity.shape}"
        )

    top_k = None if args.top_k_candidates is None or args.top_k_candidates <= 0 else int(args.top_k_candidates)
    det_tensors = load_detection_tensors(
        scene_root,
        args.detection_source,
        top_k_candidates=top_k,
        camera_filter=args.camera,
    )
    cameras = load_camera_parameters(args.camera_params)
    if args.camera is not None:
        keep = set(args.camera)
        cameras = {cid: cam for cid, cam in cameras.items() if cid in keep}

    common_cams = [cid for cid in det_tensors.keys() if cid in cameras]
    if not common_cams:
        raise RuntimeError("No overlapping camera ids between detections and camera parameters.")

    print("2D Candidate Evaluation Against Projected 3D GT")
    print(f"  scene_root: {scene_root}")
    print(f"  gt_3d: {args.gt_3d} ({'json' if args.gt_3d.suffix.lower()=='.json' else 'npz'})")
    if args.gt_3d.suffix.lower() == ".npz":
        print(f"  gt_key: {args.gt_key}")
        if args.gt_validity_npz is not None:
            print(f"  gt_validity_npz: {args.gt_validity_npz} (key={args.gt_validity_key})")
    print(f"  detection_source: {args.detection_source}")
    print(f"  compare_space: {args.compare_space}")
    print(f"  hand_swap: {args.hand_swap}")
    if image_resolution is not None and image_diagonal_px is not None:
        width, height = image_resolution
        print(
            f"  image_resolution: {width:g}x{height:g} "
            f"(diagonal_px={_format_float(image_diagonal_px)})"
        )
        diag_thresholds = ", ".join(f"{t:g}%" for t in args.diag_pck_thresholds_percent)
        print(f"  diag_pck_thresholds: {diag_thresholds}")
    else:
        print("  [warn] image_resolution unavailable; diagonal-normalized metrics disabled")
    hand_thresholds = ", ".join(f"{t:g}%" for t in args.hand_pck_thresholds_percent)
    print(
        f"  hand_normalization: GT bbox longest side "
        f"(min_joints={args.hand_scale_min_joints}, min_scale_px={args.hand_scale_min_px:g})"
    )
    print(f"  hand_pck_thresholds: {hand_thresholds}")
    if args.exclude_joints:
        print(f"  exclude_joints: {', '.join(str(j) for j in args.exclude_joints)}")
    if args.exclude_frames:
        print(f"  exclude_frames: {len(args.exclude_frames)}")
    print(f"  cameras: {len(common_cams)} ({', '.join(common_cams)})")
    print(f"  gt_shape: frames={gt_positions.shape[0]}, joints={gt_positions.shape[1]}")
    if frame_keys is not None:
        head = ", ".join(frame_keys[:3]) if frame_keys else ""
        tail = ", ".join(frame_keys[-3:]) if frame_keys else ""
        print(f"  json_frame_keys: first=[{head}] last=[{tail}]")

    per_camera_stats: Dict[str, CandidateEvalStats] = {}
    overall = CandidateEvalStats()
    for cam_id in common_cams:
        cam = cameras[cam_id]
        det = det_tensors[cam_id]
        cam_stats = evaluate_camera(
            cam_id=cam_id,
            cam=cam,
            det_tensor=det,
            gt_positions=gt_positions,
            gt_validity=gt_validity,
            det_valid_threshold=float(args.det_valid_threshold),
            compare_undistorted=(args.compare_space == "undistorted"),
            hand_scale_min_joints=int(args.hand_scale_min_joints),
            hand_scale_min_px=float(args.hand_scale_min_px),
            hand_swap_mode=str(args.hand_swap),
            exclude_frames=args.exclude_frames,
            exclude_joints=args.exclude_joints,
        )
        per_camera_stats[cam_id] = cam_stats
        overall.merge(cam_stats)

        det_frames = det.shape[0]
        eval_frames = min(gt_positions.shape[0], det_frames)
        if det_frames != gt_positions.shape[0]:
            print(
                f"  [warn] camera {cam_id}: gt_frames={gt_positions.shape[0]} "
                f"det_frames={det_frames}, evaluated min={eval_frames}"
            )

    _print_stats(
        "Overall",
        overall,
        args.pck_thresholds,
        image_diagonal_px=image_diagonal_px,
        diag_thresholds_percent=args.diag_pck_thresholds_percent,
        hand_thresholds_percent=args.hand_pck_thresholds_percent,
    )
    for cam_id in common_cams:
        _print_stats(
            f"Camera {cam_id}",
            per_camera_stats[cam_id],
            args.pck_thresholds,
            image_diagonal_px=image_diagonal_px,
            diag_thresholds_percent=args.diag_pck_thresholds_percent,
            hand_thresholds_percent=args.hand_pck_thresholds_percent,
        )


if __name__ == "__main__":
    main()
