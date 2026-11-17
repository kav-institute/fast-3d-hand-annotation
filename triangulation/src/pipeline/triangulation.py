"""
Robust multi-view triangulation with candidate sampling and debug support.
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import repeat
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None
from .dataloading import CameraModel
from .geometry import (
    epipolar_distance,
    homogenize,
    huber_weight,
    project_point,
    triangulate_linear,
    weighted_triangulate,
)


@dataclass
class Candidate2D:
    xy: np.ndarray
    score: float
    spread: float


@dataclass
class _CandidateGrid:
    points: np.ndarray
    origin: np.ndarray
    cell_size: float
    cells: Dict[Tuple[int, int], List[int]]
    bounds: Tuple[float, float, float, float]


def _point_line_distance_static(line: np.ndarray, point: np.ndarray) -> float:
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    denom = math.hypot(a, b)
    if denom < 1e-9:
        return float("inf")
    return abs(a * float(point[0]) + b * float(point[1]) + c) / denom


def _line_box_segment_static(
    line: np.ndarray, bounds: Tuple[float, float, float, float]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    min_x, max_x, min_y, max_y = bounds
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    points: List[np.ndarray] = []
    if abs(b) > 1e-9:
        for x in (min_x, max_x):
            y = -(a * x + c) / b
            if min_y <= y <= max_y:
                points.append(np.array([x, y], dtype=np.float64))
    if abs(a) > 1e-9:
        for y in (min_y, max_y):
            x = -(b * y + c) / a
            if min_x <= x <= max_x:
                points.append(np.array([x, y], dtype=np.float64))
    if len(points) < 2:
        return None
    if len(points) > 2:
        best_pair = (points[0], points[1])
        best_dist = -1.0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = float(np.linalg.norm(points[i] - points[j]))
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (points[i], points[j])
        return best_pair
    return points[0], points[1]


def _min_dist_to_line_in_grid_static(
    line: np.ndarray, grid: _CandidateGrid, grid_radius: int
) -> Optional[float]:
    segment = _line_box_segment_static(line, grid.bounds)
    if segment is None:
        return None
    p0, p1 = segment
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        sample_pts = np.array([p0], dtype=np.float64)
    else:
        steps = max(2, int(length / grid.cell_size) + 1)
        ts = np.linspace(0.0, 1.0, steps)
        sample_pts = p0[None, :] + ts[:, None] * delta[None, :]
    seen: set[int] = set()
    best = float("inf")
    for x, y in sample_pts:
        gx = int((x - grid.origin[0]) // grid.cell_size)
        gy = int((y - grid.origin[1]) // grid.cell_size)
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                key = (gx + dx, gy + dy)
                if key not in grid.cells:
                    continue
                for idx in grid.cells[key]:
                    if idx in seen:
                        continue
                    seen.add(idx)
                    pt = grid.points[idx]
                    dist = _point_line_distance_static(line, pt)
                    if dist < best:
                        best = dist
    if not np.isfinite(best):
        return None
    return float(best)


def _candidate_grid_weights_for_camera(
    cam_i: str,
    cands_i: List[Candidate2D],
    cam_ids: Sequence[str],
    grids: Dict[str, Optional[_CandidateGrid]],
    fundamentals: Dict[Tuple[str, str], np.ndarray],
    epi_sigma_px: float,
    grid_radius: int,
) -> Tuple[str, List[float]]:
    weights_i: List[float] = []
    if not cands_i:
        return cam_i, weights_i
    for cand_i in cands_i:
        dists: List[float] = []
        x_i = homogenize(cand_i.xy)
        for cam_j in cam_ids:
            if cam_j == cam_i:
                continue
            grid_j = grids.get(cam_j)
            if grid_j is None:
                continue
            F_ij = fundamentals[(cam_i, cam_j)]
            line = F_ij @ x_i
            best = _min_dist_to_line_in_grid_static(line, grid_j, grid_radius)
            if best is not None:
                dists.append(float(best))
        if not dists:
            epi_w = 1.0
        else:
            avg_dist = float(np.mean(dists))
            epi_w = math.exp(-(avg_dist ** 2) / (2.0 * epi_sigma_px ** 2))
        weights_i.append(epi_w)
    return cam_i, weights_i


@dataclass
class JointObservation:
    camera_id: str
    xy: np.ndarray
    pixel_sigma: float
    valid: bool = True
    candidates: List[Candidate2D] = field(default_factory=list)


@dataclass
class TriangulationResult:
    point: Optional[np.ndarray]
    inlier_cameras: List[str]
    reprojection_errors: Dict[str, float]
    weights: Dict[str, float]
    residual_sigma: float
    selected_candidates: Dict[str, Candidate2D] = field(default_factory=dict)
    debug: Dict[str, object] = field(default_factory=dict)

    def as_measurement(self) -> Optional[np.ndarray]:
        if self.point is None:
            return None
        return self.point[:3]


def compute_view_weights(
    joint_measurements: Dict[str, JointObservation],
    fundamentals: Dict[Tuple[str, str], np.ndarray],
    epi_sigma_px: float,
) -> Dict[str, float]:
    return _compute_view_weights_internal(joint_measurements, fundamentals, epi_sigma_px, False)[0]


def compute_view_weights_with_debug(
    joint_measurements: Dict[str, JointObservation],
    fundamentals: Dict[Tuple[str, str], np.ndarray],
    epi_sigma_px: float,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
    return _compute_view_weights_internal(joint_measurements, fundamentals, epi_sigma_px, True)


def _compute_view_weights_internal(
    joint_measurements: Dict[str, JointObservation],
    fundamentals: Dict[Tuple[str, str], np.ndarray],
    epi_sigma_px: float,
    want_debug: bool,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
    weights: Dict[str, float] = {}
    debug_info: Dict[str, Dict[str, object]] = {}
    cam_ids = list(joint_measurements.keys())
    for cam_i in cam_ids:
        obs_i = joint_measurements[cam_i]
        if not obs_i.valid:
            if want_debug:
                debug_info[cam_i] = {"pairwise_epi": {}, "mean_epi": None, "reliability": 0.0}
            continue
        accum: List[float] = []
        for cam_j in cam_ids:
            if cam_i == cam_j:
                continue
            obs_j = joint_measurements[cam_j]
            if not obs_j.valid:
                continue
            F_ij = fundamentals[(cam_i, cam_j)]
            dist = epipolar_distance(obs_i.xy, obs_j.xy, F_ij)
            accum.append(dist)
            if want_debug:
                info = debug_info.setdefault(
                    cam_i, {"pairwise_epi": {}, "mean_epi": None, "reliability": None}
                )
                info["pairwise_epi"][cam_j] = float(dist)
        if not accum:
            weights[cam_i] = 1.0
            if want_debug:
                info = debug_info.setdefault(
                    cam_i, {"pairwise_epi": {}, "mean_epi": None, "reliability": None}
                )
                info["mean_epi"] = None
        else:
            avg_dist = float(np.mean(accum))
            weights[cam_i] = math.exp(-(avg_dist ** 2) / (2.0 * epi_sigma_px ** 2))
            if want_debug:
                info = debug_info.setdefault(
                    cam_i, {"pairwise_epi": {}, "mean_epi": None, "reliability": None}
                )
                info["mean_epi"] = float(avg_dist)
    if not weights:
        return weights, debug_info
    max_w = max(weights.values())
    if max_w < 1e-6:
        return weights, debug_info
    for cam_id in list(weights.keys()):
        obs = joint_measurements[cam_id]
        reliability = 1.0 / max(obs.pixel_sigma, 1.0)
        weights[cam_id] = (weights[cam_id] / max_w) * reliability
        if want_debug:
            info = debug_info.setdefault(
                cam_id, {"pairwise_epi": {}, "mean_epi": None, "reliability": None}
            )
            info["reliability"] = float(reliability)
    norm = sum(weights.values())
    if norm > 1e-6:
        scale = len(weights) / norm
        for cam_id in weights:
            weights[cam_id] *= scale
            if want_debug:
                info = debug_info.setdefault(
                    cam_id, {"pairwise_epi": {}, "mean_epi": None, "reliability": None}
                )
                info["normalized_weight"] = float(weights[cam_id])
    if want_debug:
        for cam_id in weights:
            info = debug_info.setdefault(
                cam_id, {"pairwise_epi": {}, "mean_epi": None, "reliability": None}
            )
            info.setdefault("normalized_weight", float(weights[cam_id]))
    return weights, debug_info


class Triangulator:
    def __init__(
        self,
        cameras: Dict[str, CameraModel],
        fundamentals: Dict[Tuple[str, str], np.ndarray],
        reproj_thresh_px: float = 3.0,
        epi_sigma_px: float = 2.0,
        min_inliers: int = 2,
        ransac_iters: int = 128,
        ransac_workers: int = 1,
        ransac_progress: bool = False,
        ransac_executor: str = "thread",
        baseline_thresh_deg: float = 2.0,
        subset_size: int = 2,
        rng: Optional[random.Random] = None,
        debug_enabled: bool = False,
        include_aggregated_candidate: bool = True,
        epi_weighting_mode: str = "aggregated",
        epi_grid_size_px: float = 20.0,
        epi_grid_radius: int = 1,
        epi_grid_workers: int = 1,
    ) -> None:
        self.cameras = cameras
        self.fundamentals = fundamentals
        self.reproj_thresh = reproj_thresh_px
        self.epi_sigma_px = epi_sigma_px
        self.min_inliers = min_inliers
        self.ransac_iters = ransac_iters
        self.ransac_workers = max(1, ransac_workers)
        self.ransac_progress = ransac_progress
        self.ransac_executor = ransac_executor
        self.baseline_thresh_deg = baseline_thresh_deg
        self.ransac_subset_size = max(2, subset_size)
        self.rng = rng or random.Random()
        self.debug_enabled = debug_enabled
        self.include_aggregated_candidate = include_aggregated_candidate
        if epi_weighting_mode not in {"none", "aggregated", "candidates", "candidates_grid"}:
            raise ValueError(
                "epi_weighting_mode must be one of {'none', 'aggregated', 'candidates', 'candidates_grid'}."
            )
        self.epi_weighting_mode = epi_weighting_mode
        self.epi_grid_size_px = max(1.0, float(epi_grid_size_px))
        self.epi_grid_radius = max(0, int(epi_grid_radius))
        self.epi_grid_workers = max(1, int(epi_grid_workers))

    def triangulate_joint(
        self,
        observations: Dict[str, JointObservation],
    ) -> TriangulationResult:
        valid_obs = {cid: obs for cid, obs in observations.items() if obs.valid}
        candidate_options: Dict[str, List[Candidate2D]] = {}
        for cid, obs in list(valid_obs.items()):
            cand_list = self._candidate_list_from_observation(obs)
            if not cand_list:
                valid_obs.pop(cid)
                continue
            candidate_options[cid] = cand_list
        if len(valid_obs) < 2:
            debug_info = {}
            if self.debug_enabled and self.epi_weighting_mode != "none":
                _, weight_debug = compute_view_weights_with_debug(
                    valid_obs, self.fundamentals, self.epi_sigma_px
                )
                debug_info["weights_debug"] = weight_debug
                debug_info["candidate_options"] = self._serialize_candidate_options(candidate_options)
                debug_info["reason"] = "insufficient_views"
            return TriangulationResult(
                None, [], {}, {}, residual_sigma=float("inf"), selected_candidates={}, debug=debug_info
            )
        if self.epi_weighting_mode == "none":
            weights = {cid: 1.0 for cid in valid_obs}
            weight_debug = {}
        elif self.epi_weighting_mode == "candidates":
            candidate_options, weights, weight_debug = self._apply_candidate_epipolar_weights(
                candidate_options, valid_obs
            )
        elif self.epi_weighting_mode == "candidates_grid":
            candidate_options, weights, weight_debug = self._apply_candidate_epipolar_weights_grid(
                candidate_options, valid_obs
            )
        else:
            if self.debug_enabled:
                weights, weight_debug = compute_view_weights_with_debug(
                    valid_obs, self.fundamentals, self.epi_sigma_px
                )
            else:
                weights = compute_view_weights(valid_obs, self.fundamentals, self.epi_sigma_px)
                weight_debug = {}
        cam_ids = list(valid_obs.keys())
        best_point: Optional[np.ndarray] = None
        best_inliers: List[str] = []
        best_score = -1.0
        best_errors: Dict[str, float] = {}
        best_candidate_map: Dict[str, Candidate2D] = {}

        subset_k = min(self.ransac_subset_size, len(cam_ids))
        if subset_k < 2:
            subset_k = 2
        if self.ransac_workers <= 1:
            iter_range = range(self.ransac_iters)
            if self.ransac_progress and tqdm is not None:
                iter_range = tqdm(iter_range, total=self.ransac_iters, desc="RANSAC", leave=False)
            for _ in iter_range:
                subset = self.rng.sample(cam_ids, k=subset_k)
                P_subset = [self.cameras[cid].P for cid in subset]
                x_subset = []
                for cid in subset:
                    chosen = self._sample_candidate(candidate_options[cid], rng=self.rng)
                    x_subset.append(chosen.xy)
                candidate = triangulate_linear(P_subset, x_subset)
                errors, candidate_map = self._compute_candidate_residuals(
                    candidate, valid_obs, candidate_options
                )
                inliers = [cid for cid, err in errors.items() if err < self.reproj_thresh]
                if len(inliers) < self.min_inliers:
                    continue
                score = sum(
                    weights.get(cid, 1.0) * self._candidate_weight(candidate_map.get(cid))
                    for cid in inliers
                    if candidate_map.get(cid) is not None
                )
                if score > best_score:
                    best_score = score
                    best_point = candidate
                    best_inliers = inliers
                    best_errors = errors
                    best_candidate_map = candidate_map
        elif self.ransac_executor == "process":
            camera_Ps = {cid: self.cameras[cid].P for cid in cam_ids}
            batch_count = min(self.ransac_workers, self.ransac_iters)
            base, rem = divmod(self.ransac_iters, batch_count)
            batch_sizes = [base + (1 if idx < rem else 0) for idx in range(batch_count)]
            seeds = [self.rng.randrange(2**32 - 1) for _ in range(batch_count)]
            results_iter = None
            with ProcessPoolExecutor(max_workers=self.ransac_workers) as executor:
                results_iter = executor.map(
                    _ransac_batch_worker,
                    seeds,
                    batch_sizes,
                    repeat(cam_ids),
                    repeat(subset_k),
                    repeat(camera_Ps),
                    repeat(candidate_options),
                    repeat(valid_obs),
                    repeat(weights),
                    repeat(self.reproj_thresh),
                    repeat(self.min_inliers),
                )
                if self.ransac_progress and tqdm is not None:
                    pbar = tqdm(total=self.ransac_iters, desc="RANSAC", leave=False)
                else:
                    pbar = None
                for batch_size, result in zip(batch_sizes, results_iter):
                    if pbar is not None:
                        pbar.update(batch_size)
                    if result is None:
                        continue
                    score, candidate, inliers, errors, candidate_map = result
                    if score > best_score:
                        best_score = score
                        best_point = candidate
                        best_inliers = inliers
                        best_errors = errors
                        best_candidate_map = candidate_map
                if pbar is not None:
                    pbar.close()
        else:
            seeds = [self.rng.randrange(2**32 - 1) for _ in range(self.ransac_iters)]
            worker = partial(
                self._ransac_trial,
                cam_ids=cam_ids,
                subset_k=subset_k,
                candidate_options=candidate_options,
                observations=valid_obs,
                weights=weights,
            )
            with ThreadPoolExecutor(max_workers=self.ransac_workers) as executor:
                results_iter = executor.map(worker, seeds)
                if self.ransac_progress and tqdm is not None:
                    results_iter = tqdm(
                        results_iter, total=self.ransac_iters, desc="RANSAC", leave=False
                    )
                for result in results_iter:
                    if result is None:
                        continue
                    score, candidate, inliers, errors, candidate_map = result
                    if score > best_score:
                        best_score = score
                        best_point = candidate
                        best_inliers = inliers
                        best_errors = errors
                        best_candidate_map = candidate_map
        if best_point is None:
            debug_info = {}
            if self.debug_enabled:
                debug_info = {
                    "weights": weights,
                    "weights_debug": weight_debug,
                    "candidate_options": self._serialize_candidate_options(candidate_options),
                    "reason": "ransac_failed",
                }
            return TriangulationResult(
                None, [], {}, {}, residual_sigma=float("inf"), selected_candidates={}, debug=debug_info
            )
        refined, refined_candidate_map = self._refine(
            best_point,
            best_inliers,
            valid_obs,
            weights,
            candidate_options,
            best_candidate_map,
        )
        baseline_angle = self._compute_baseline_angle(refined, best_inliers)
        if baseline_angle < self.baseline_thresh_deg:
            debug_info = {}
            if self.debug_enabled:
                debug_info = {
                    "weights": weights,
                    "weights_debug": weight_debug,
                    "point_linear": best_point.tolist(),
                    "baseline_angle_deg": float(baseline_angle),
                    "candidate_options": self._serialize_candidate_options(candidate_options),
                    "selected_candidates": self._serialize_candidate_map(refined_candidate_map),
                    "reason": "baseline_too_small",
                }
            return TriangulationResult(
                None, [], {}, {}, residual_sigma=float("inf"), selected_candidates={}, debug=debug_info
            )
        residuals, final_candidate_map = self._compute_candidate_residuals(
            refined, valid_obs, candidate_options
        )
        sigma = np.median([residuals[cid] for cid in best_inliers]) if best_inliers else float("inf")
        debug_info = {}
        if self.debug_enabled:
            debug_info = {
                "weights": weights,
                "weights_debug": weight_debug,
                "best_inliers": best_inliers,
                "initial_point": best_point.tolist(),
                "refined_point": refined.tolist(),
                "best_errors": best_errors,
                "residuals": residuals,
                "baseline_angle_deg": float(baseline_angle),
                "residual_sigma": float(sigma),
                "candidate_options": self._serialize_candidate_options(candidate_options),
                "selected_candidates": self._serialize_candidate_map(final_candidate_map),
            }
        return TriangulationResult(
            refined,
            best_inliers,
            residuals,
            weights,
            residual_sigma=sigma,
            selected_candidates=final_candidate_map,
            debug=debug_info,
        )

    def triangulate_joint_marginal(
        self,
        observations: Dict[str, JointObservation],
        em_iters: int = 3,
        sigma_floor: float = 1.0,
        log_eps: float = 1e-9,
    ) -> TriangulationResult:
        valid_obs = {cid: obs for cid, obs in observations.items() if obs.valid}
        candidate_options: Dict[str, List[Candidate2D]] = {}
        for cid, obs in list(valid_obs.items()):
            cand_list = self._candidate_list_from_observation(obs)
            if not cand_list:
                valid_obs.pop(cid)
                continue
            candidate_options[cid] = cand_list
        if len(valid_obs) < 2:
            debug_info = {}
            if self.debug_enabled and self.epi_weighting_mode != "none":
                _, weight_debug = compute_view_weights_with_debug(
                    valid_obs, self.fundamentals, self.epi_sigma_px
                )
                debug_info["weights_debug"] = weight_debug
                debug_info["candidate_options"] = self._serialize_candidate_options(candidate_options)
                debug_info["reason"] = "insufficient_views"
            return TriangulationResult(
                None, [], {}, {}, residual_sigma=float("inf"), selected_candidates={}, debug=debug_info
            )

        if self.epi_weighting_mode == "none":
            weights = {cid: 1.0 for cid in valid_obs}
            weight_debug = {}
        elif self.epi_weighting_mode == "candidates":
            candidate_options, weights, weight_debug = self._apply_candidate_epipolar_weights(
                candidate_options, valid_obs
            )
        elif self.epi_weighting_mode == "candidates_grid":
            candidate_options, weights, weight_debug = self._apply_candidate_epipolar_weights_grid(
                candidate_options, valid_obs
            )
        else:
            if self.debug_enabled:
                weights, weight_debug = compute_view_weights_with_debug(
                    valid_obs, self.fundamentals, self.epi_sigma_px
                )
            else:
                weights = compute_view_weights(valid_obs, self.fundamentals, self.epi_sigma_px)
                weight_debug = {}

        cam_ids = list(valid_obs.keys())
        best_point: Optional[np.ndarray] = None
        best_score = -float("inf")

        subset_k = min(self.ransac_subset_size, len(cam_ids))
        if subset_k < 2:
            subset_k = 2
        if self.ransac_workers <= 1:
            iter_range = range(self.ransac_iters)
            if self.ransac_progress and tqdm is not None:
                iter_range = tqdm(iter_range, total=self.ransac_iters, desc="RANSAC", leave=False)
            for _ in iter_range:
                subset = self.rng.sample(cam_ids, k=subset_k)
                P_subset = [self.cameras[cid].P for cid in subset]
                x_subset = []
                for cid in subset:
                    chosen = self._sample_candidate(candidate_options[cid], rng=self.rng)
                    x_subset.append(chosen.xy)
                candidate = triangulate_linear(P_subset, x_subset)
                score, support = self._soft_score(
                    candidate,
                    valid_obs,
                    candidate_options,
                    weights,
                    sigma_floor=sigma_floor,
                    log_eps=log_eps,
                )
                if support < self.min_inliers:
                    continue
                if score > best_score:
                    best_score = score
                    best_point = candidate
        elif self.ransac_executor == "process":
            camera_Ps = {cid: self.cameras[cid].P for cid in cam_ids}
            batch_count = min(self.ransac_workers, self.ransac_iters)
            base, rem = divmod(self.ransac_iters, batch_count)
            batch_sizes = [base + (1 if idx < rem else 0) for idx in range(batch_count)]
            seeds = [self.rng.randrange(2**32 - 1) for _ in range(batch_count)]
            with ProcessPoolExecutor(max_workers=self.ransac_workers) as executor:
                results_iter = executor.map(
                    _soft_ransac_batch_worker,
                    seeds,
                    batch_sizes,
                    repeat(cam_ids),
                    repeat(subset_k),
                    repeat(camera_Ps),
                    repeat(candidate_options),
                    repeat(valid_obs),
                    repeat(weights),
                    repeat(float(sigma_floor)),
                    repeat(float(log_eps)),
                    repeat(self.min_inliers),
                )
                if self.ransac_progress and tqdm is not None:
                    pbar = tqdm(total=self.ransac_iters, desc="RANSAC", leave=False)
                else:
                    pbar = None
                for batch_size, result in zip(batch_sizes, results_iter):
                    if pbar is not None:
                        pbar.update(batch_size)
                    if result is None:
                        continue
                    score, candidate = result
                    if score > best_score:
                        best_score = score
                        best_point = candidate
                if pbar is not None:
                    pbar.close()
        else:
            seeds = [self.rng.randrange(2**32 - 1) for _ in range(self.ransac_iters)]
            worker = partial(
                self._soft_ransac_trial,
                cam_ids=cam_ids,
                subset_k=subset_k,
                candidate_options=candidate_options,
                observations=valid_obs,
                weights=weights,
                sigma_floor=sigma_floor,
                log_eps=log_eps,
            )
            with ThreadPoolExecutor(max_workers=self.ransac_workers) as executor:
                results_iter = executor.map(worker, seeds)
                if self.ransac_progress and tqdm is not None:
                    results_iter = tqdm(
                        results_iter, total=self.ransac_iters, desc="RANSAC", leave=False
                    )
                for result in results_iter:
                    if result is None:
                        continue
                    score, candidate = result
                    if score > best_score:
                        best_score = score
                        best_point = candidate

        if best_point is None:
            debug_info = {}
            if self.debug_enabled:
                debug_info = {
                    "weights": weights,
                    "weights_debug": weight_debug,
                    "candidate_options": self._serialize_candidate_options(candidate_options),
                    "reason": "ransac_failed",
                }
            return TriangulationResult(
                None, [], {}, {}, residual_sigma=float("inf"), selected_candidates={}, debug=debug_info
            )

        refined = self._soft_refine(
            best_point,
            valid_obs,
            candidate_options,
            weights,
            em_iters=em_iters,
            sigma_floor=sigma_floor,
        )
        support_views = list(valid_obs.keys())
        baseline_angle = self._compute_baseline_angle(refined, support_views)
        if baseline_angle < self.baseline_thresh_deg:
            debug_info = {}
            if self.debug_enabled:
                debug_info = {
                    "weights": weights,
                    "weights_debug": weight_debug,
                    "point_linear": best_point.tolist(),
                    "baseline_angle_deg": float(baseline_angle),
                    "candidate_options": self._serialize_candidate_options(candidate_options),
                    "reason": "baseline_too_small",
                }
            return TriangulationResult(
                None, [], {}, {}, residual_sigma=float("inf"), selected_candidates={}, debug=debug_info
            )

        residuals, final_candidate_map = self._compute_candidate_residuals(
            refined, valid_obs, candidate_options
        )
        sigma = np.median([residuals[cid] for cid in support_views]) if support_views else float("inf")
        debug_info = {}
        if self.debug_enabled:
            debug_info = {
                "weights": weights,
                "weights_debug": weight_debug,
                "best_score": float(best_score),
                "refined_point": refined.tolist(),
                "residuals": residuals,
                "baseline_angle_deg": float(baseline_angle),
                "residual_sigma": float(sigma),
                "candidate_options": self._serialize_candidate_options(candidate_options),
                "selected_candidates": self._serialize_candidate_map(final_candidate_map),
            }
        return TriangulationResult(
            refined,
            support_views,
            residuals,
            weights,
            residual_sigma=sigma,
            selected_candidates=final_candidate_map,
            debug=debug_info,
        )

    def _compute_candidate_residuals(
        self,
        X: np.ndarray,
        observations: Dict[str, JointObservation],
        candidate_options: Dict[str, List[Candidate2D]],
    ) -> Tuple[Dict[str, float], Dict[str, Candidate2D]]:
        errors: Dict[str, float] = {}
        best_candidates: Dict[str, Candidate2D] = {}
        for cam_id, obs in observations.items():
            P = self.cameras[cam_id].P
            proj = project_point(P, X)
            best_err = float("inf")
            best_cand: Optional[Candidate2D] = None
            for cand in candidate_options.get(cam_id, []):
                err = float(np.linalg.norm(proj - cand.xy))
                if err < best_err:
                    best_err = err
                    best_cand = cand
            if best_cand is None and np.all(np.isfinite(obs.xy)):
                fallback = Candidate2D(obs.xy, score=1.0, spread=float(obs.pixel_sigma))
                best_cand = fallback
                best_err = float(np.linalg.norm(proj - obs.xy))
            errors[cam_id] = best_err
            if best_cand is not None:
                best_candidates[cam_id] = best_cand
        return errors, best_candidates

    def _refine(
        self,
        initial: np.ndarray,
        inliers: List[str],
        observations: Dict[str, JointObservation],
        weights: Dict[str, float],
        candidate_options: Dict[str, List[Candidate2D]],
        initial_candidates: Dict[str, Candidate2D],
        huber_delta: float = 4.0,
        iterations: int = 3,
    ) -> Tuple[np.ndarray, Dict[str, Candidate2D]]:
        X = initial.copy()
        candidate_map = initial_candidates.copy()
        for _ in range(iterations):
            projections = []
            points = []
            wts = []
            errors, candidate_map = self._compute_candidate_residuals(X, observations, candidate_options)
            for cam_id in inliers:
                candidate = candidate_map.get(cam_id)
                if candidate is None:
                    continue
                projections.append(self.cameras[cam_id].P)
                points.append(candidate.xy)
                err = errors.get(cam_id, float("inf"))
                base_w = weights.get(cam_id, 1.0) * self._candidate_weight(candidate)
                wts.append(base_w * huber_weight(err, huber_delta))
            if not wts:
                break
            X = weighted_triangulate(projections, points, wts)
        return X, candidate_map

    def _candidate_list_from_observation(self, obs: JointObservation) -> List[Candidate2D]:
        candidates = list(obs.candidates)
        if self.include_aggregated_candidate and np.all(np.isfinite(obs.xy)):
            aggregated = Candidate2D(obs.xy, score=1.0, spread=float(obs.pixel_sigma))
            if not any(np.allclose(c.xy, aggregated.xy) for c in candidates):
                candidates.append(aggregated)
        return candidates

    def _soft_score(
        self,
        X: np.ndarray,
        observations: Dict[str, JointObservation],
        candidate_options: Dict[str, List[Candidate2D]],
        weights: Dict[str, float],
        sigma_floor: float,
        log_eps: float,
    ) -> Tuple[float, int]:
        score = 0.0
        support = 0
        for cam_id, obs in observations.items():
            candidates = candidate_options.get(cam_id, [])
            if not candidates:
                continue
            proj = project_point(self.cameras[cam_id].P, X)
            if not np.all(np.isfinite(proj)):
                continue
            support += 1
            sigma = max(float(obs.pixel_sigma), float(sigma_floor))
            denom = 2.0 * sigma * sigma
            l_sum = 0.0
            for cand in candidates:
                p = self._candidate_weight(cand)
                d2 = float(np.sum((proj - cand.xy) ** 2))
                if not np.isfinite(d2):
                    continue
                l_sum += p * math.exp(-d2 / denom)
            score += math.log(log_eps + weights.get(cam_id, 1.0) * l_sum)
        return score, support

    def _soft_refine(
        self,
        initial: np.ndarray,
        observations: Dict[str, JointObservation],
        candidate_options: Dict[str, List[Candidate2D]],
        weights: Dict[str, float],
        em_iters: int,
        sigma_floor: float,
    ) -> np.ndarray:
        X = initial.copy()
        for _ in range(max(1, int(em_iters))):
            projections: List[np.ndarray] = []
            points: List[np.ndarray] = []
            wts: List[float] = []
            for cam_id, obs in observations.items():
                candidates = candidate_options.get(cam_id, [])
                if not candidates:
                    continue
                proj = project_point(self.cameras[cam_id].P, X)
                if not np.all(np.isfinite(proj)):
                    continue
                sigma = max(float(obs.pixel_sigma), float(sigma_floor))
                denom = 2.0 * sigma * sigma
                resp: List[float] = []
                total = 0.0
                for cand in candidates:
                    p = self._candidate_weight(cand)
                    d2 = float(np.sum((proj - cand.xy) ** 2))
                    if not np.isfinite(d2):
                        resp.append(0.0)
                        continue
                    r = p * math.exp(-d2 / denom)
                    if not np.isfinite(r):
                        resp.append(0.0)
                        continue
                    resp.append(r)
                    total += r
                if not np.isfinite(total) or total < 1e-12:
                    continue
                inv_total = 1.0 / total
                view_w = weights.get(cam_id, 1.0)
                for cand, r in zip(candidates, resp):
                    r_norm = r * inv_total
                    w = view_w * r_norm
                    if (not np.isfinite(w)) or w <= 1e-12:
                        continue
                    projections.append(self.cameras[cam_id].P)
                    points.append(cand.xy)
                    wts.append(w)
            if len(wts) < 2:
                break
            try:
                X_new = weighted_triangulate(projections, points, wts)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(X_new)):
                break
            X = X_new
        return X

    def _soft_ransac_trial(
        self,
        seed: int,
        cam_ids: Sequence[str],
        subset_k: int,
        candidate_options: Dict[str, List[Candidate2D]],
        observations: Dict[str, JointObservation],
        weights: Dict[str, float],
        sigma_floor: float,
        log_eps: float,
    ) -> Optional[Tuple[float, np.ndarray]]:
        rng = random.Random(seed)
        subset = rng.sample(list(cam_ids), k=subset_k)
        P_subset = [self.cameras[cid].P for cid in subset]
        x_subset = []
        for cid in subset:
            chosen = self._sample_candidate(candidate_options[cid], rng=rng)
            x_subset.append(chosen.xy)
        try:
            candidate = triangulate_linear(P_subset, x_subset)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(candidate)):
            return None
        score, support = self._soft_score(
            candidate, observations, candidate_options, weights, sigma_floor, log_eps
        )
        if support < self.min_inliers:
            return None
        return score, candidate
    def _apply_candidate_epipolar_weights(
        self,
        candidate_options: Dict[str, List[Candidate2D]],
        observations: Dict[str, JointObservation],
    ) -> Tuple[Dict[str, List[Candidate2D]], Dict[str, float], Dict[str, Dict[str, object]]]:
        cam_ids = list(candidate_options.keys())
        if not cam_ids:
            return candidate_options, {}, {}
        per_cam_weights: Dict[str, List[float]] = {}
        for cam_i in cam_ids:
            cands_i = candidate_options.get(cam_i, [])
            weights_i: List[float] = []
            for cand_i in cands_i:
                dists: List[float] = []
                for cam_j in cam_ids:
                    if cam_j == cam_i:
                        continue
                    cands_j = candidate_options.get(cam_j, [])
                    if not cands_j:
                        continue
                    F_ij = self.fundamentals[(cam_i, cam_j)]
                    best = min(
                        epipolar_distance(cand_i.xy, cand_j.xy, F_ij) for cand_j in cands_j
                    )
                    dists.append(float(best))
                if not dists:
                    epi_w = 1.0
                else:
                    avg_dist = float(np.mean(dists))
                    epi_w = math.exp(-(avg_dist ** 2) / (2.0 * self.epi_sigma_px ** 2))
                weights_i.append(epi_w)
            per_cam_weights[cam_i] = weights_i

        weighted_candidates: Dict[str, List[Candidate2D]] = {}
        for cam_id, cands in candidate_options.items():
            weights_i = per_cam_weights.get(cam_id, [])
            updated: List[Candidate2D] = []
            for idx, cand in enumerate(cands):
                epi_w = weights_i[idx] if idx < len(weights_i) else 1.0
                updated.append(Candidate2D(cand.xy, score=cand.score * epi_w, spread=cand.spread))
            weighted_candidates[cam_id] = updated

        view_weights: Dict[str, float] = {}
        debug_info: Dict[str, Dict[str, object]] = {}
        for cam_id in cam_ids:
            wts = per_cam_weights.get(cam_id, [])
            if wts:
                avg_w = float(np.mean(wts))
            else:
                avg_w = 1.0
            view_weights[cam_id] = avg_w
            if self.debug_enabled:
                debug_info[cam_id] = {
                    "candidate_epi_weights": [float(w) for w in wts],
                    "mean_candidate_weight": float(avg_w),
                }
        if view_weights:
            max_w = max(view_weights.values())
            if max_w >= 1e-6:
                for cam_id in list(view_weights.keys()):
                    reliability = 1.0 / max(observations[cam_id].pixel_sigma, 1.0)
                    view_weights[cam_id] = (view_weights[cam_id] / max_w) * reliability
                    if self.debug_enabled:
                        info = debug_info.setdefault(cam_id, {})
                        info["reliability"] = float(reliability)
            norm = sum(view_weights.values())
            if norm > 1e-6:
                scale = len(view_weights) / norm
                for cam_id in view_weights:
                    view_weights[cam_id] *= scale
                    if self.debug_enabled:
                        info = debug_info.setdefault(cam_id, {})
                        info["normalized_weight"] = float(view_weights[cam_id])
        return weighted_candidates, view_weights, debug_info

    def _apply_candidate_epipolar_weights_grid(
        self,
        candidate_options: Dict[str, List[Candidate2D]],
        observations: Dict[str, JointObservation],
    ) -> Tuple[Dict[str, List[Candidate2D]], Dict[str, float], Dict[str, Dict[str, object]]]:
        grids: Dict[str, Optional[_CandidateGrid]] = {}
        for cam_id, cands in candidate_options.items():
            grids[cam_id] = self._build_candidate_grid(cands, self.epi_grid_size_px)
        cam_ids = list(candidate_options.keys())
        if not cam_ids:
            return candidate_options, {}, {}

        per_cam_weights: Dict[str, List[float]] = {}
        if self.epi_grid_workers > 1:
            with ProcessPoolExecutor(max_workers=self.epi_grid_workers) as executor:
                cam_list = list(cam_ids)
                cands_list = [candidate_options.get(cam_i, []) for cam_i in cam_list]
                for cam_id, weights_i in executor.map(
                    _candidate_grid_weights_for_camera,
                    cam_list,
                    cands_list,
                    repeat(cam_ids),
                    repeat(grids),
                    repeat(self.fundamentals),
                    repeat(self.epi_sigma_px),
                    repeat(self.epi_grid_radius),
                ):
                    per_cam_weights[cam_id] = weights_i
        else:
            for cam_i in cam_ids:
                cands_i = candidate_options.get(cam_i, [])
                cam_id, weights_i = _candidate_grid_weights_for_camera(
                    cam_i,
                    cands_i,
                    cam_ids,
                    grids,
                    self.fundamentals,
                    self.epi_sigma_px,
                    self.epi_grid_radius,
                )
                per_cam_weights[cam_id] = weights_i

        weighted_candidates: Dict[str, List[Candidate2D]] = {}
        for cam_id, cands in candidate_options.items():
            weights_i = per_cam_weights.get(cam_id, [])
            updated: List[Candidate2D] = []
            for idx, cand in enumerate(cands):
                epi_w = weights_i[idx] if idx < len(weights_i) else 1.0
                updated.append(Candidate2D(cand.xy, score=cand.score * epi_w, spread=cand.spread))
            weighted_candidates[cam_id] = updated

        view_weights: Dict[str, float] = {}
        debug_info: Dict[str, Dict[str, object]] = {}
        for cam_id in cam_ids:
            wts = per_cam_weights.get(cam_id, [])
            if wts:
                avg_w = float(np.mean(wts))
            else:
                avg_w = 1.0
            view_weights[cam_id] = avg_w
            if self.debug_enabled:
                debug_info[cam_id] = {
                    "candidate_epi_weights": [float(w) for w in wts],
                    "mean_candidate_weight": float(avg_w),
                }
        if view_weights:
            max_w = max(view_weights.values())
            if max_w >= 1e-6:
                for cam_id in list(view_weights.keys()):
                    reliability = 1.0 / max(observations[cam_id].pixel_sigma, 1.0)
                    view_weights[cam_id] = (view_weights[cam_id] / max_w) * reliability
                    if self.debug_enabled:
                        info = debug_info.setdefault(cam_id, {})
                        info["reliability"] = float(reliability)
            norm = sum(view_weights.values())
            if norm > 1e-6:
                scale = len(view_weights) / norm
        for cam_id in view_weights:
            view_weights[cam_id] *= scale
            if self.debug_enabled:
                info = debug_info.setdefault(cam_id, {})
                info["normalized_weight"] = float(view_weights[cam_id])
        return weighted_candidates, view_weights, debug_info

    def _build_candidate_grid(
        self,
        candidates: List[Candidate2D],
        cell_size: float,
    ) -> Optional[_CandidateGrid]:
        if not candidates:
            return None
        pts = np.array([c.xy for c in candidates], dtype=np.float64)
        if pts.size == 0:
            return None
        min_x = float(np.min(pts[:, 0]))
        max_x = float(np.max(pts[:, 0]))
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        margin = float(cell_size)
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin
        origin = np.array([min_x, min_y], dtype=np.float64)
        cells: Dict[Tuple[int, int], List[int]] = {}
        for idx, (x, y) in enumerate(pts):
            gx = int((x - origin[0]) // cell_size)
            gy = int((y - origin[1]) // cell_size)
            cells.setdefault((gx, gy), []).append(idx)
        return _CandidateGrid(
            points=pts,
            origin=origin,
            cell_size=float(cell_size),
            cells=cells,
            bounds=(min_x, max_x, min_y, max_y),
        )

    def _min_dist_to_line_in_grid(self, line: np.ndarray, grid: _CandidateGrid) -> Optional[float]:
        segment = self._line_box_segment(line, grid.bounds)
        if segment is None:
            return None
        p0, p1 = segment
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            sample_pts = np.array([p0], dtype=np.float64)
        else:
            steps = max(2, int(length / grid.cell_size) + 1)
            ts = np.linspace(0.0, 1.0, steps)
            sample_pts = p0[None, :] + ts[:, None] * delta[None, :]
        seen: set[int] = set()
        best = float("inf")
        for x, y in sample_pts:
            gx = int((x - grid.origin[0]) // grid.cell_size)
            gy = int((y - grid.origin[1]) // grid.cell_size)
            for dx in range(-self.epi_grid_radius, self.epi_grid_radius + 1):
                for dy in range(-self.epi_grid_radius, self.epi_grid_radius + 1):
                    key = (gx + dx, gy + dy)
                    if key not in grid.cells:
                        continue
                    for idx in grid.cells[key]:
                        if idx in seen:
                            continue
                        seen.add(idx)
                        pt = grid.points[idx]
                        dist = self._point_line_distance(line, pt)
                        if dist < best:
                            best = dist
        if not np.isfinite(best):
            return None
        return float(best)

    @staticmethod
    def _point_line_distance(line: np.ndarray, point: np.ndarray) -> float:
        a, b, c = float(line[0]), float(line[1]), float(line[2])
        denom = math.hypot(a, b)
        if denom < 1e-9:
            return float("inf")
        return abs(a * float(point[0]) + b * float(point[1]) + c) / denom

    @staticmethod
    def _line_box_segment(
        line: np.ndarray, bounds: Tuple[float, float, float, float]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        min_x, max_x, min_y, max_y = bounds
        a, b, c = float(line[0]), float(line[1]), float(line[2])
        points: List[np.ndarray] = []
        if abs(b) > 1e-9:
            for x in (min_x, max_x):
                y = -(a * x + c) / b
                if min_y <= y <= max_y:
                    points.append(np.array([x, y], dtype=np.float64))
        if abs(a) > 1e-9:
            for y in (min_y, max_y):
                x = -(b * y + c) / a
                if min_x <= x <= max_x:
                    points.append(np.array([x, y], dtype=np.float64))
        if len(points) < 2:
            return None
        if len(points) > 2:
            best_pair = (points[0], points[1])
            best_dist = -1.0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = float(np.linalg.norm(points[i] - points[j]))
                    if dist > best_dist:
                        best_dist = dist
                        best_pair = (points[i], points[j])
            return best_pair
        return points[0], points[1]

    def _sample_candidate(
        self, candidates: List[Candidate2D], rng: Optional[random.Random] = None
    ) -> Candidate2D:
        if not candidates:
            raise ValueError("No candidates available for sampling")
        if len(candidates) == 1:
            return candidates[0]
        rng = rng or self.rng
        weights = [max(self._candidate_weight(c), 1e-6) for c in candidates]
        total = sum(weights)
        r = rng.random() * total
        cumulative = 0.0
        for cand, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return cand
        return candidates[-1]

    def _ransac_trial(
        self,
        seed: int,
        cam_ids: Sequence[str],
        subset_k: int,
        candidate_options: Dict[str, List[Candidate2D]],
        observations: Dict[str, JointObservation],
        weights: Dict[str, float],
    ) -> Optional[
        Tuple[float, np.ndarray, List[str], Dict[str, float], Dict[str, Candidate2D]]
    ]:
        rng = random.Random(seed)
        subset = rng.sample(list(cam_ids), k=subset_k)
        P_subset = [self.cameras[cid].P for cid in subset]
        x_subset = []
        for cid in subset:
            chosen = self._sample_candidate(candidate_options[cid], rng=rng)
            x_subset.append(chosen.xy)
        candidate = triangulate_linear(P_subset, x_subset)
        errors, candidate_map = self._compute_candidate_residuals(
            candidate, observations, candidate_options
        )
        inliers = [cid for cid, err in errors.items() if err < self.reproj_thresh]
        if len(inliers) < self.min_inliers:
            return None
        score = sum(
            weights.get(cid, 1.0) * self._candidate_weight(candidate_map.get(cid))
            for cid in inliers
            if candidate_map.get(cid) is not None
        )
        return score, candidate, inliers, errors, candidate_map

    @staticmethod
    def _candidate_weight(candidate: Optional[Candidate2D]) -> float:
        if candidate is None:
            return 1.0
        return max(candidate.score, 1e-3) / (1.0 + candidate.spread)

    def _compute_baseline_angle(self, X: np.ndarray, inliers: List[str]) -> float:
        if len(inliers) < 2:
            return 0.0
        rays: List[np.ndarray] = []
        for cam_id in inliers:
            cam = self.cameras[cam_id]
            ray = X[:3] - cam.center.reshape(3)
            norm = np.linalg.norm(ray)
            if norm < 1e-6:
                continue
            rays.append(ray / norm)
        if len(rays) < 2:
            return 0.0
        max_angle = 0.0
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                angle = math.degrees(math.acos(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0)))
                max_angle = max(max_angle, angle)
        return max_angle

    @staticmethod
    def _serialize_candidate_map(candidate_map: Dict[str, Candidate2D]) -> Dict[str, Dict[str, float]]:
        return {
            cam_id: {
                "xy": [float(c.xy[0]), float(c.xy[1])],
                "score": float(c.score),
                "spread": float(c.spread),
            }
            for cam_id, c in candidate_map.items()
        }

    @staticmethod
    def _serialize_candidate_options(
        candidate_options: Dict[str, List[Candidate2D]]
    ) -> Dict[str, List[Dict[str, float]]]:
        return {
            cam_id: [
                {
                    "xy": [float(c.xy[0]), float(c.xy[1])],
                    "score": float(c.score),
                    "spread": float(c.spread),
                }
                for c in cand_list
            ]
            for cam_id, cand_list in candidate_options.items()
        }


def _candidate_weight_value(candidate: Optional[Candidate2D]) -> float:
    if candidate is None:
        return 1.0
    return max(candidate.score, 1e-3) / (1.0 + candidate.spread)


def _sample_candidate_local(candidates: List[Candidate2D], rng: random.Random) -> Candidate2D:
    if not candidates:
        raise ValueError("No candidates available for sampling")
    if len(candidates) == 1:
        return candidates[0]
    weights = [max(_candidate_weight_value(c), 1e-6) for c in candidates]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for cand, w in zip(candidates, weights):
        cumulative += w
        if r <= cumulative:
            return cand
    return candidates[-1]


def _compute_candidate_residuals_local(
    camera_Ps: Dict[str, np.ndarray],
    X: np.ndarray,
    observations: Dict[str, JointObservation],
    candidate_options: Dict[str, List[Candidate2D]],
) -> Tuple[Dict[str, float], Dict[str, Candidate2D]]:
    errors: Dict[str, float] = {}
    best_candidates: Dict[str, Candidate2D] = {}
    for cam_id, obs in observations.items():
        P = camera_Ps[cam_id]
        proj = project_point(P, X)
        best_err = float("inf")
        best_cand: Optional[Candidate2D] = None
        for cand in candidate_options.get(cam_id, []):
            err = float(np.linalg.norm(proj - cand.xy))
            if err < best_err:
                best_err = err
                best_cand = cand
        if best_cand is None and np.all(np.isfinite(obs.xy)):
            fallback = Candidate2D(obs.xy, score=1.0, spread=float(obs.pixel_sigma))
            best_cand = fallback
            best_err = float(np.linalg.norm(proj - obs.xy))
        errors[cam_id] = best_err
        if best_cand is not None:
            best_candidates[cam_id] = best_cand
    return errors, best_candidates


def _ransac_batch_worker(
    seed: int,
    iterations: int,
    cam_ids: Sequence[str],
    subset_k: int,
    camera_Ps: Dict[str, np.ndarray],
    candidate_options: Dict[str, List[Candidate2D]],
    observations: Dict[str, JointObservation],
    weights: Dict[str, float],
    reproj_thresh: float,
    min_inliers: int,
) -> Optional[Tuple[float, np.ndarray, List[str], Dict[str, float], Dict[str, Candidate2D]]]:
    rng = random.Random(seed)
    best_score = -1.0
    best_point: Optional[np.ndarray] = None
    best_inliers: List[str] = []
    best_errors: Dict[str, float] = {}
    best_candidate_map: Dict[str, Candidate2D] = {}
    cam_ids_list = list(cam_ids)
    for _ in range(iterations):
        subset = rng.sample(cam_ids_list, k=subset_k)
        P_subset = [camera_Ps[cid] for cid in subset]
        x_subset = []
        for cid in subset:
            chosen = _sample_candidate_local(candidate_options[cid], rng=rng)
            x_subset.append(chosen.xy)
        candidate = triangulate_linear(P_subset, x_subset)
        errors, candidate_map = _compute_candidate_residuals_local(
            camera_Ps, candidate, observations, candidate_options
        )
        inliers = [cid for cid, err in errors.items() if err < reproj_thresh]
        if len(inliers) < min_inliers:
            continue
        score = sum(
            weights.get(cid, 1.0) * _candidate_weight_value(candidate_map.get(cid))
            for cid in inliers
            if candidate_map.get(cid) is not None
        )
        if score > best_score:
            best_score = score
            best_point = candidate
            best_inliers = inliers
            best_errors = errors
            best_candidate_map = candidate_map
    if best_point is None:
        return None
    return best_score, best_point, best_inliers, best_errors, best_candidate_map


def _soft_score_candidate_local(
    camera_Ps: Dict[str, np.ndarray],
    X: np.ndarray,
    observations: Dict[str, JointObservation],
    candidate_options: Dict[str, List[Candidate2D]],
    weights: Dict[str, float],
    sigma_floor: float,
    log_eps: float,
) -> Tuple[float, int]:
    score = 0.0
    support = 0
    for cam_id, obs in observations.items():
        candidates = candidate_options.get(cam_id, [])
        if not candidates:
            continue
        proj = project_point(camera_Ps[cam_id], X)
        if not np.all(np.isfinite(proj)):
            continue
        support += 1
        sigma = max(float(obs.pixel_sigma), float(sigma_floor))
        denom = 2.0 * sigma * sigma
        l_sum = 0.0
        for cand in candidates:
            p = _candidate_weight_value(cand)
            d2 = float(np.sum((proj - cand.xy) ** 2))
            if not np.isfinite(d2):
                continue
            l_sum += p * math.exp(-d2 / denom)
        score += math.log(log_eps + weights.get(cam_id, 1.0) * l_sum)
    return score, support


def _soft_ransac_batch_worker(
    seed: int,
    iterations: int,
    cam_ids: Sequence[str],
    subset_k: int,
    camera_Ps: Dict[str, np.ndarray],
    candidate_options: Dict[str, List[Candidate2D]],
    observations: Dict[str, JointObservation],
    weights: Dict[str, float],
    sigma_floor: float,
    log_eps: float,
    min_inliers: int,
) -> Optional[Tuple[float, np.ndarray]]:
    rng = random.Random(seed)
    best_score = -float("inf")
    best_point: Optional[np.ndarray] = None
    cam_ids_list = list(cam_ids)
    for _ in range(iterations):
        subset = rng.sample(cam_ids_list, k=subset_k)
        P_subset = [camera_Ps[cid] for cid in subset]
        x_subset = []
        for cid in subset:
            chosen = _sample_candidate_local(candidate_options[cid], rng=rng)
            x_subset.append(chosen.xy)
        try:
            candidate = triangulate_linear(P_subset, x_subset)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(candidate)):
            continue
        score, support = _soft_score_candidate_local(
            camera_Ps,
            candidate,
            observations,
            candidate_options,
            weights,
            sigma_floor,
            log_eps,
        )
        if support < min_inliers:
            continue
        if score > best_score:
            best_score = score
            best_point = candidate
    if best_point is None:
        return None
    return best_score, best_point


def _project_with_jacobian(
    P: np.ndarray,
    X: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    X_h = np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    x = float(P[0] @ X_h)
    y = float(P[1] @ X_h)
    z = float(P[2] @ X_h)
    if abs(z) < 1e-9:
        return None
    u = x / z
    v = y / z
    p1 = P[0, :3]
    p2 = P[1, :3]
    p3 = P[2, :3]
    denom = z * z
    du_dX = (p1 * z - p3 * x) / denom
    dv_dX = (p2 * z - p3 * y) / denom
    J = np.vstack([du_dX, dv_dX])
    return np.array([u, v], dtype=np.float64), J


def triangulate_point_ba(
    observations: Dict[str, JointObservation],
    cameras: Dict[str, CameraModel],
    max_iters: int = 10,
    huber_delta: float = 4.0,
) -> Tuple[Optional[np.ndarray], float]:
    valid_items: List[Tuple[str, JointObservation]] = []
    for cam_id, obs in observations.items():
        if not obs.valid:
            continue
        if not np.all(np.isfinite(obs.xy)):
            continue
        if cam_id not in cameras:
            continue
        valid_items.append((cam_id, obs))
    if len(valid_items) < 2:
        return None, float("inf")

    projections = [cameras[cam_id].P for cam_id, _ in valid_items]
    points = [obs.xy for _, obs in valid_items]
    X_h = triangulate_linear(projections, points)
    if abs(X_h[3]) > 1e-9:
        X = X_h[:3] / X_h[3]
    else:
        X = X_h[:3].copy()

    for _ in range(max(1, int(max_iters))):
        J_rows: List[np.ndarray] = []
        r_rows: List[np.ndarray] = []
        for cam_id, obs in valid_items:
            result = _project_with_jacobian(cameras[cam_id].P, X)
            if result is None:
                continue
            proj, J = result
            r = obs.xy - proj
            sigma = max(float(obs.pixel_sigma), 1.0)
            err = float(np.linalg.norm(r))
            norm_err = err / sigma
            w = huber_weight(norm_err, huber_delta) / (sigma * sigma)
            if not np.isfinite(w) or w <= 0.0:
                continue
            sqrt_w = math.sqrt(w)
            J_rows.append(sqrt_w * J)
            r_rows.append(sqrt_w * r)
        if len(J_rows) < 2:
            break
        J_all = np.vstack(J_rows)
        r_all = np.concatenate(r_rows)
        step, *_ = np.linalg.lstsq(J_all, r_all, rcond=None)
        if not np.all(np.isfinite(step)):
            break
        X = X + step
        if float(np.linalg.norm(step)) < 1e-6:
            break

    errors: List[float] = []
    X_h_final = np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    for cam_id, obs in valid_items:
        proj = project_point(cameras[cam_id].P, X_h_final)
        errors.append(float(np.linalg.norm(obs.xy - proj)))
    residual_sigma = float(np.median(errors)) if errors else float("inf")
    return X_h_final, residual_sigma
