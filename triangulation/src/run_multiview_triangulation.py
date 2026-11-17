#!/usr/bin/env python3
"""Minimal multi-view triangulation runner for method comparisons."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

from pipeline.dataloading import CameraModel, load_camera_parameters, load_detection_tensor
from pipeline.geometry import compute_fundamental_from_projections
from pipeline.postprocess import GraphOptConfig, run_post_process
from pipeline.preprocessing import aggregate_candidates, undistort_points
from pipeline.triangulation import (
    Candidate2D,
    JointObservation,
    Triangulator,
    triangulate_point_ba,
)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    epi_weighting_mode: str
    include_aggregated_candidate: bool
    top_k_candidates: Optional[int]
    per_view_candidates: Optional[int]
    candidate_source: str
    solver: str


METHODS: Dict[str, MethodConfig] = {
    "baseline": MethodConfig(
        name="baseline",
        epi_weighting_mode="none",
        include_aggregated_candidate=False,
        top_k_candidates=1,
        per_view_candidates=0,
        candidate_source="raw",
        solver="ransac",
    ),
    "candidates_grid": MethodConfig(
        name="candidates_grid",
        epi_weighting_mode="candidates_grid",
        include_aggregated_candidate=False,
        top_k_candidates=100,
        per_view_candidates=0,
        candidate_source="aggregate",
        solver="ransac",
    ),
    "aggregated": MethodConfig(
        name="aggregated",
        epi_weighting_mode="aggregated",
        include_aggregated_candidate=True,
        top_k_candidates=100,
        per_view_candidates=0,
        candidate_source="aggregate",
        solver="ransac",
    ),
    "aggregated_ba": MethodConfig(
        name="aggregated_ba",
        epi_weighting_mode="none",
        include_aggregated_candidate=True,
        top_k_candidates=100,
        per_view_candidates=0,
        candidate_source="aggregate",
        solver="ba",
    ),
    "soft_marginal": MethodConfig(
        name="soft_marginal",
        epi_weighting_mode="aggregated",
        include_aggregated_candidate=False,
        top_k_candidates=0,
        per_view_candidates=0,
        candidate_source="aggregate",
        solver="soft_marginal",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal multi-view triangulation.")
    parser.add_argument(
        "--detection-source-path",
        type=Path,
        required=True,
        help="Scene root folder containing camera parameters and detections.",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        required=True,
        choices=sorted(METHODS.keys()),
        help="Which triangulation method(s) to run.",
    )
    parser.add_argument(
        "--detection-source",
        nargs="+",
        default=["default"],
        choices=["default", "mediapipe", "sapiens"],
        help="Detection source naming scheme(s).",
    )
    parser.add_argument("--ransac-iters", type=int, default=5000)
    parser.add_argument("--ransac-subset-size", type=int, default=2)
    parser.add_argument("--min-inliers", type=int, default=2)
    parser.add_argument("--ransac-workers", type=int, default=32)
    parser.add_argument("--ransac-executor", choices=["thread", "process"], default="process")
    parser.add_argument("--ransac-progress", action="store_true")
    parser.add_argument("--reproj-thresh-px", type=float, default=5.0)
    parser.add_argument("--baseline-thresh-deg", type=float, default=2.0)
    parser.add_argument("--epi-sigma-px", type=float, default=2.0)
    parser.add_argument("--epi-grid-size-px", type=float, default=20.0)
    parser.add_argument("--epi-grid-radius", type=int, default=1)
    parser.add_argument("--epi-grid-workers", type=int, default=32)
    parser.add_argument(
        "--run-post",
        action="store_true",
        help="Run post-processing (refine + graph optimization) after triangulation.",
    )
    parser.add_argument(
        "--post-only",
        action="store_true",
        help="Run only post-processing on existing outputs_{source}_{method} folders.",
    )
    parser.add_argument(
        "--post-graph-window",
        type=int,
        default=10,
        help="Graph optimization window size (frames) for post-processing.",
    )
    parser.add_argument(
        "--post-graph-overlap",
        type=int,
        default=5,
        help="Graph optimization overlap (frames) for post-processing.",
    )
    parser.add_argument(
        "--soft-em-iters",
        type=int,
        default=3,
        help="EM refinement iterations for soft_marginal.",
    )
    parser.add_argument(
        "--soft-sigma-floor",
        type=float,
        default=1.0,
        help="Sigma floor (px) for soft_marginal likelihood.",
    )
    parser.add_argument(
        "--soft-log-eps",
        type=float,
        default=1e-9,
        help="Log-epsilon to stabilize soft_marginal scoring.",
    )
    parser.add_argument("--ba-iters", type=int, default=10, help="Iterations for aggregated_ba.")
    parser.add_argument(
        "--ba-huber-delta",
        type=float,
        default=4.0,
        help="Huber delta for aggregated_ba (applied to residual/sigma).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable overall progress bar/ETA.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output directory root (defaults to detection-source-path).",
    )
    return parser.parse_args()


def _load_param_file(scene_root: Path) -> Path:
    matches = list(scene_root.glob("camera_parameter*.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one camera parameter file in {scene_root}, found: {matches}"
        )
    return matches[0]


def _load_all_detections(scene_root: Path, detection_source: str) -> Dict[str, np.ndarray]:
    detections: Dict[str, np.ndarray] = {}

    subd = ""
    if detection_source == "mediapipe":
        subd = "_mp"

    elif detection_source == "sapiens":
        subd = "_sp"


    for det_path in sorted(scene_root.glob(f"detections{subd}_camera_*_2d.npy")):
        cam_id = det_path.name[len(f"detections{subd}_camera_"):-len("_2d.npy")]
        detections[cam_id] = load_detection_tensor(det_path, top_k=None)
    if not detections:
        raise RuntimeError(
            f"No detection tensors found for source '{detection_source}' in {scene_root}"
        )
    shapes = {(a.shape[0],a.shape[2]) for a in detections.values()}
    if len(shapes) != 1:
        raise ValueError(f"Camera detections must have identical frame/joint counts, got {shapes}")
    if len(detections) < 2:
        raise ValueError("Triangulation requires detections from at least two calibrated cameras")
    return detections


def _precompute_fundamentals(cameras: Dict[str, CameraModel]) -> Dict[tuple[str, str], np.ndarray]:
    ids = list(cameras.keys())
    fundamentals: Dict[tuple[str, str], np.ndarray] = {}
    for i in ids:
        for j in ids:
            if i == j:
                continue
            fundamentals[(i, j)] = compute_fundamental_from_projections(cameras[i].P, cameras[j].P)
    return fundamentals


def _slice_candidates(candidates: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    if top_k is None or top_k <= 0:
        return candidates
    if top_k >= candidates.shape[0]:
        return candidates
    return candidates[:top_k]


def _build_observations_for_frame(
    frame_idx: int,
    det_per_view: Dict[str, np.ndarray],
    cameras: Dict[str, CameraModel],
    num_joints: int,
    method: MethodConfig,
) -> Dict[str, List[JointObservation]]:
    joint_obs_per_view: Dict[str, List[JointObservation]] = {}
    for cam_id, tensor in det_per_view.items():
        candidates = tensor[frame_idx]
        candidates = _slice_candidates(candidates, method.top_k_candidates)
        camera = cameras[cam_id]

        if method.candidate_source == "raw":
            obs_list: List[JointObservation] = []
            for j in range(num_joints):
                valid_mask = candidates[:, j, 3] > 0.5
                pts = candidates[valid_mask, j, :2]
                if pts.size == 0:
                    obs_list.append(
                        JointObservation(cam_id, np.zeros(2), 1.0, valid=False, candidates=[])
                    )
                    continue
                undist_pts = undistort_points(pts, camera.K, camera.dist_coeff)
                cand_list = [
                    Candidate2D(xy=undist_pts[idx], score=1.0, spread=0.0)
                    for idx in range(len(undist_pts))
                ]
                obs_list.append(
                    JointObservation(cam_id, np.zeros(2), 1.0, valid=True, candidates=cand_list)
                )
            joint_obs_per_view[cam_id] = obs_list
            continue

        max_candidates = method.per_view_candidates
        if max_candidates is not None and max_candidates <= 0:
            max_candidates = None
        xy, sigma, raw_candidates = aggregate_candidates(
            candidates,
            max_candidates=max_candidates,
            cluster_radius_px=None,
            density_bandwidth_px=None,
            density_nms_radius_px=None,
            aggregate_mode="robust",
        )
        converted_xy = xy.copy()
        if not np.isnan(converted_xy).all():
            converted_xy = undistort_points(converted_xy, camera.K, camera.dist_coeff)

        per_joint_candidates: List[List[Candidate2D]] = [[] for _ in range(num_joints)]
        for j in range(num_joints):
            joint_candidates = raw_candidates[j]
            if not joint_candidates:
                continue
            pts = np.array([entry["xy"] for entry in joint_candidates], dtype=np.float64)
            undist_pts = undistort_points(pts, camera.K, camera.dist_coeff)
            for idx, entry in enumerate(joint_candidates):
                per_joint_candidates[j].append(
                    Candidate2D(
                        xy=undist_pts[idx],
                        score=float(entry["score"]),
                        spread=float(entry["spread"]),
                    )
                )

        valid_mask = np.isfinite(converted_xy[:, 0]) & np.isfinite(converted_xy[:, 1])
        obs_list = []
        for j in range(num_joints):
            sigma_value = float(sigma[j]) if not np.isnan(sigma[j]) else float("inf")
            if not valid_mask[j]:
                obs_list.append(
                    JointObservation(
                        cam_id,
                        np.zeros(2),
                        sigma_value,
                        valid=False,
                        candidates=per_joint_candidates[j],
                    )
                )
            else:
                obs_list.append(
                    JointObservation(
                        cam_id,
                        converted_xy[j],
                        sigma_value,
                        valid=True,
                        candidates=per_joint_candidates[j],
                    )
                )
        joint_obs_per_view[cam_id] = obs_list
    return joint_obs_per_view


def _run_method(
    scene_root: Path,
    cameras: Dict[str, CameraModel],
    fundamentals: Dict[tuple[str, str], np.ndarray],
    det_per_view: Dict[str, np.ndarray],
    method: MethodConfig,
    args: argparse.Namespace,
    detection_source: str,
) -> None:
    frames = min(tensor.shape[0] for tensor in det_per_view.values())
    num_joints = next(iter(det_per_view.values())).shape[2]

    triangulator = None
    if method.solver not in {"ba"}:
        triangulator = Triangulator(
            cameras,
            fundamentals,
            reproj_thresh_px=args.reproj_thresh_px,
            epi_sigma_px=args.epi_sigma_px,
            min_inliers=args.min_inliers,
            ransac_iters=args.ransac_iters,
            ransac_workers=args.ransac_workers,
            ransac_progress=args.ransac_progress,
            ransac_executor=args.ransac_executor,
            baseline_thresh_deg=args.baseline_thresh_deg,
            subset_size=args.ransac_subset_size,
            rng=random.Random(0),
            debug_enabled=False,
            include_aggregated_candidate=method.include_aggregated_candidate,
            epi_weighting_mode=method.epi_weighting_mode,
            epi_grid_size_px=args.epi_grid_size_px,
            epi_grid_radius=args.epi_grid_radius,
            epi_grid_workers=args.epi_grid_workers,
        )

    positions_raw = np.full((frames, num_joints, 3), np.nan, dtype=np.float64)
    validity = np.zeros((frames, num_joints), dtype=bool)
    residual_sigmas = np.full((frames, num_joints), np.inf, dtype=np.float64)

    total_steps = frames * num_joints
    use_progress = tqdm is not None and not args.no_progress
    pbar = None
    if use_progress:
        pbar = tqdm(
            total=total_steps,
            desc=f"{detection_source}/{method.name}",
            leave=False,
        )
    start = time.perf_counter()
    for f in range(frames):
        joint_obs_per_view = _build_observations_for_frame(
            f, det_per_view, cameras, num_joints, method
        )
        for j in range(num_joints):
            joint_measurements = {cam_id: obs_list[j] for cam_id, obs_list in joint_obs_per_view.items()}
            if method.solver == "ba":
                point, residual = triangulate_point_ba(
                    joint_measurements,
                    cameras,
                    max_iters=args.ba_iters,
                    huber_delta=args.ba_huber_delta,
                )
                if point is not None:
                    positions_raw[f, j] = point[:3]
                    validity[f, j] = True
                residual_sigmas[f, j] = residual
            elif method.solver == "soft_marginal":
                if triangulator is None:
                    raise RuntimeError("Triangulator not initialized for soft_marginal method.")
                result = triangulator.triangulate_joint_marginal(
                    joint_measurements,
                    em_iters=args.soft_em_iters,
                    sigma_floor=args.soft_sigma_floor,
                    log_eps=args.soft_log_eps,
                )
                if result.point is not None:
                    positions_raw[f, j] = result.point[:3]
                    validity[f, j] = True
                residual_sigmas[f, j] = result.residual_sigma
            else:
                if triangulator is None:
                    raise RuntimeError("Triangulator not initialized for RANSAC method.")
                result = triangulator.triangulate_joint(joint_measurements)
                if result.point is not None:
                    positions_raw[f, j] = result.point[:3]
                    validity[f, j] = True
                residual_sigmas[f, j] = result.residual_sigma
            if pbar is not None:
                pbar.update(1)
    elapsed = time.perf_counter() - start
    if pbar is not None:
        pbar.close()

    output_root = args.output_root or scene_root
    output_dir = output_root / f"outputs_{detection_source}_{method.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    label_state_path = output_dir / "triangulation_label_stage.npz"
    output_path = output_dir / "triangulation_pipeline_outputs.npz"
    np.savez_compressed(
        output_path,
        positions_raw=positions_raw,
        validity=validity,
        residual_sigmas=residual_sigmas,
    )
    np.savez_compressed(
        label_state_path,
        positions_raw=positions_raw,
        validity=validity,
        residual_sigmas=residual_sigmas,
        manual_annotations=np.zeros_like(validity, dtype=bool),
    )
    print(
        f"Saved {method.name} ({detection_source}) to {output_path} "
        f"({elapsed} s)"
    )
    if args.run_post:
        _run_post_stage(
            output_dir,
            label_state_path,
            args,
        )


def _run_post_stage(
    output_dir: Path,
    label_state_path: Path,
    args: argparse.Namespace,
) -> None:
    if not label_state_path.exists():
        print(f"Post stage skipped; label state not found at {label_state_path}")
        return

    with np.load(label_state_path, allow_pickle=True) as data:
        positions_raw = np.asarray(data["positions_raw"], dtype=np.float64)
        validity = np.asarray(data["validity"], dtype=bool)
        residual_sigmas = np.asarray(data["residual_sigmas"], dtype=np.float64)
        manual_annotations_arr = data.get("manual_annotations")

    if manual_annotations_arr is None or np.shape(manual_annotations_arr) != np.shape(validity):
        manual_annotations = np.zeros_like(validity, dtype=bool)
    else:
        try:
            manual_annotations = np.asarray(manual_annotations_arr, dtype=bool)
        except (TypeError, ValueError):
            manual_annotations = np.zeros_like(validity, dtype=bool)

    graph_cfg = GraphOptConfig(
        measurement_weight=1.0,
        smooth_weight=0.3,
        bone_weight=1.0,
        max_iterations=20,
        verbose=True,
    )
    results = run_post_process(
        positions_raw=positions_raw,
        validity=validity,
        residual_sigmas=residual_sigmas,
        manual_annotations=manual_annotations,
        lambda_vel=0.2,
        graph_window=args.post_graph_window,
        graph_overlap=args.post_graph_overlap,
        graph_config=graph_cfg,
    )
    np.savez_compressed(output_dir / "triangulation_pipeline_outputs.npz", **results)
    print(f"Post stage complete for {output_dir}")


def main() -> None:
    args = parse_args()
    scene_root = args.detection_source_path
    param_file = _load_param_file(scene_root)

    if args.post_only:
        output_root = args.output_root or scene_root
        for det_source in args.detection_source:
            for method_name in args.method:
                method = METHODS[method_name]
                output_dir = output_root / f"outputs_{det_source}_{method.name}"
                label_state_path = output_dir / "triangulation_label_stage.npz"
                _run_post_stage(
                    output_dir,
                    label_state_path,
                    args,
                )
        return

    cameras = load_camera_parameters(param_file)
    fundamentals = _precompute_fundamentals(cameras)

    for det_source in args.detection_source:
        det_per_view = _load_all_detections(scene_root, det_source)
        unknown = set(det_per_view) - set(cameras)
        if unknown:
            raise ValueError(f"Missing calibration for detection cameras: {sorted(unknown)}")
        for method_name in args.method:
            method = METHODS[method_name]
            _run_method(
                scene_root,
                cameras,
                fundamentals,
                det_per_view,
                method,
                args,
                det_source,
            )


if __name__ == "__main__":
    main()
