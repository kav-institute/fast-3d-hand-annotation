#!/usr/bin/env python3
"""
Pairwise PCK evaluation between triangulation outputs or JSON GT.

Example:

    python evaluate_pck_json.py \
        --results scene_a/outputs/triangulation_pipeline_outputs.npz \
        --gt scene_b/outputs/triangulation_pipeline_outputs.npz \
        --key positions_refined \
        --pck-thresholds 5 10 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None

from pipeline.visualization import load_skeleton


def _ensure_interactive_backend() -> bool:
    """Try to switch matplotlib to an interactive backend if currently on Agg."""
    if plt is None:
        return False
    backend = str(plt.get_backend()).lower()
    if "agg" not in backend:
        return True
    for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "MacOSX", "GTK3Agg"):
        try:
            plt.switch_backend(candidate)
            return True
        except Exception:
            continue
    print(f"matplotlib backend '{backend}' is non-interactive; cannot show plot. "
          "Install a GUI backend or set MPLBACKEND (e.g., TkAgg/Qt5Agg).")
    return False


def load_positions_npz(path: Path, key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")
    positions = np.asarray(data[key], dtype=np.float64)
    validity = data.get("validity")
    if validity is None or validity.shape[:2] != positions.shape[:2]:
        valid_mask = np.isfinite(positions).all(axis=2)
    else:
        valid_mask = np.asarray(validity, dtype=bool)
    return positions, valid_mask


def _infer_label(result_path: Path) -> str:
    for parent in result_path.parents:
        name = parent.name
        if name.startswith("outputs_"):
            return name
        if name == "outputs":
            return name
    return result_path.stem


def _sort_frame_keys(keys: Sequence[str]) -> List[str]:
    def key_sort_value(k: str) -> Tuple[int, object]:
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)
    return sorted(keys, key=key_sort_value)


def load_positions_json(
    path: Path,
    frame_start: Optional[int],
    frame_count: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object of frames in {path}")
    keys = _sort_frame_keys(list(data.keys()))
    if not keys:
        raise ValueError(f"No frames found in {path}")
    start_idx = 0
    if frame_start is not None:
        start_key = str(frame_start)
        if start_key in data:
            start_idx = keys.index(start_key)
        else:
            start_idx = int(frame_start)
    if start_idx < 0 or start_idx >= len(keys):
        raise ValueError(
            f"Frame start {frame_start} yields invalid index {start_idx} for {len(keys)} frames."
        )
    if frame_count is None:
        end_idx = len(keys)
    else:
        end_idx = start_idx + int(frame_count)
    if end_idx > len(keys):
        raise ValueError(
            f"Requested {frame_count} frames from index {start_idx}, but only {len(keys)} available."
        )
    selected = keys[start_idx:end_idx]
    frames: List[np.ndarray] = []
    valid_frames: List[np.ndarray] = []
    joint_count: Optional[int] = None
    for key in selected:
        entry = data[key]
        if isinstance(entry, dict) and "world_coord" in entry:
            frame = np.asarray(entry["world_coord"], dtype=np.float64)
            joint_valid = entry.get("joint_valid")
        else:
            frame = np.asarray(entry, dtype=np.float64)
            joint_valid = None
        if frame.ndim != 2 or frame.shape[1] != 3:
            raise ValueError(f"Frame {key} has invalid shape {frame.shape} in {path}")
        if joint_count is None:
            joint_count = frame.shape[0]
        elif frame.shape[0] != joint_count:
            raise ValueError(
                f"Inconsistent joint count in {path}: {frame.shape[0]} vs {joint_count}"
            )
        frames.append(frame)
        if isinstance(entry, dict) and entry.get("hand_type_valid") is False:
            valid_frames.append(np.zeros(frame.shape[0], dtype=bool))
        elif joint_valid is None:
            valid_frames.append(np.isfinite(frame).all(axis=1))
        else:
            valid_arr = np.asarray(joint_valid, dtype=np.float64)
            valid_arr = valid_arr.reshape(-1)
            if valid_arr.shape[0] != frame.shape[0]:
                raise ValueError(
                    f"joint_valid length {valid_arr.shape[0]} does not match joints "
                    f"{frame.shape[0]} for frame {key} in {path}"
                )
            valid_frames.append(valid_arr > 0.5)
    positions = np.stack(frames, axis=0)
    validity = np.stack(valid_frames, axis=0)
    return positions, validity, selected


def compute_pck(
    preds: np.ndarray,
    preds_valid: np.ndarray,
    gts: np.ndarray,
    gts_valid: np.ndarray,
    thresholds: Sequence[float],
    exclude_frames: Sequence[int],
    joint_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    if preds.shape != gts.shape:
        raise ValueError(f"Pred and GT shapes differ: {preds.shape} vs {gts.shape}")
    frames, joints, _ = preds.shape

    user_valid = np.array([f not in exclude_frames for f in range(frames)], dtype=bool).repeat(joints).reshape(frames, joints)

    thresholds = [float(t) for t in thresholds]

    preds_s = np.concatenate((preds[:, 21:, :], preds[:, :21, :]), axis=1)
    preds_valid_s = np.concatenate((preds_valid[:, 21:], preds_valid[:, :21]), axis=1)

    valid = preds_valid & user_valid & gts_valid & np.isfinite(preds).all(axis=2) & np.isfinite(gts).all(axis=2)
    errors_p = np.linalg.norm(preds - gts, axis=2)
    errors_p[~valid] = np.nan

    valid_s = preds_valid_s & user_valid & gts_valid & np.isfinite(preds_s).all(axis=2) & np.isfinite(gts).all(axis=2)
    errors_s = np.linalg.norm(preds_s - gts, axis=2)
    errors_s[~valid_s] = np.nan

    # Preserve the legacy GT-assisted handedness comparison, including empty frames.
    # Empty predictions must yield zero evaluated joints, not a shape error or NaN comparison.
    errors = np.full_like(errors_p, np.nan)
    for i in range(frames):
        direct = errors_p[i][np.isfinite(errors_p[i])]
        swapped = errors_s[i][np.isfinite(errors_s[i])]
        direct_mean = float(direct.mean()) if direct.size else float("inf")
        swapped_mean = float(swapped.mean()) if swapped.size else float("inf")
        errors[i] = errors_p[i] if direct_mean <= swapped_mean else errors_s[i]

    per_joint: List[Dict[str, object]] = []
    for j in range(joints):
        label = None
        if joint_names is not None and j < len(joint_names):
            label = str(joint_names[j])
        joint_errors = errors[:, j]
        finite = joint_errors[np.isfinite(joint_errors)]
        entry: Dict[str, object] = {
            "joint": j,
            "label": label,
            "count": int(finite.size),
            "mean_error": None if finite.size == 0 else float(finite.mean()),
            "median_error": None if finite.size == 0 else float(np.median(finite)),
        }
        for t in thresholds:
            key = f"pck@{t:g}"
            entry[key] = None if finite.size == 0 else float(np.mean(finite <= t))
        per_joint.append(entry)

    errors = errors[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]] #17,38
    all_errors_ = errors[np.isfinite(errors)]
    all_errors = all_errors_
    overall: Dict[str, object] = {
        "count": int(all_errors.size),
        "mean_error": None if all_errors.size == 0 else float(all_errors.mean()),
        "median_error": None if all_errors.size == 0 else float(np.median(all_errors)),
    }
    for t in thresholds:
        key = f"pck@{t:g}"
        overall[key] = None if all_errors.size == 0 else float(np.mean(all_errors <= t))

    return {
        "thresholds": thresholds,
        "overall": overall,
        "per_joint": per_joint,
        "frames": frames,
        "joints": joints,
    }


def _print_latex_table(run_name: str, metrics: Dict[str, object], table_thresholds: Sequence[float]) -> None:
    thresholds = [float(t) for t in table_thresholds]
    overall = metrics.get("overall", {})
    def format_val(val: Optional[float]) -> str:
        return "n/a" if val is None else f"{val:.3f}"

    mean_vals = [format_val(overall.get(f"pck@{t:g}")) for t in thresholds]
    print(f"{run_name} & " + " & ".join(mean_vals) + " \\\\")

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two triangulation outputs with PCK.")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        nargs="+",
        help="One or more predicted NPZs (triangulation_pipeline_outputs.npz).",
    )
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth NPZ or JSON.")
    parser.add_argument("--key", default="positions_refined", help="NPZ key to compare (default: positions_refined).")
    parser.add_argument("--gt-key", default=None, help="NPZ key for GT (defaults to --key).")
    parser.add_argument("--title", default="???", help="Plot title")
    parser.add_argument(
        "--pck-thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0],
        help="Distance thresholds for PCK (same units as the NPZ coordinates).",
    )
    parser.add_argument(
        "--exclude-frames",
        type=int,
        nargs="+",
        default=[],
        help="Exclude single frames (0-based) from evaluation.",
    )
    parser.add_argument(
        "--respect-validity",
        action="store_true",
        help="Use the validity mask from the NPZs; otherwise any finite point counts as valid.",
    )
    parser.add_argument(
        "--gt-frame-start",
        type=int,
        default=None,
        help="Frame start index or frame id for JSON GT (default 0).",
    )
    parser.add_argument(
        "--gt-frame-count",
        type=int,
        default=None,
        help="Number of frames to load from JSON GT (default matches preds).",
    )
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to dump the metrics JSON.")
    parser.add_argument(
        "--skeleton",
        type=Path,
        default=None,
        help="Optional skeleton file to supply joint names for plot/print labelling.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot overall PCK vs threshold (requires matplotlib).",
    )
    args = parser.parse_args()

    gt_key = args.gt_key or args.key
    gt_is_json = args.gt.suffix.lower() == ".json"
    gts: Optional[np.ndarray] = None
    gts_valid: Optional[np.ndarray] = None
    selected_keys: Optional[List[str]] = None
    if not gt_is_json:
        gts, gts_valid = load_positions_npz(args.gt, gt_key)
        if not args.respect_validity:
            gts_valid = np.isfinite(gts).all(axis=2)
    elif args.gt_frame_count is not None:
        gts, gts_valid, selected_keys = load_positions_json(
            args.gt,
            frame_start=args.gt_frame_start,
            frame_count=args.gt_frame_count,
        )
        if not args.respect_validity:
            gts_valid = np.isfinite(gts).all(axis=2)

    any_plots = False
    for result_path in args.results:
        preds, preds_valid = load_positions_npz(result_path, args.key)
        if not args.respect_validity:
            preds_valid = np.isfinite(preds).all(axis=2)

        if gt_is_json and args.gt_frame_count is None:
            frame_count = preds.shape[0]
            gts, gts_valid, selected_keys = load_positions_json(
                args.gt,
                frame_start=args.gt_frame_start,
                frame_count=frame_count,
            )
            if not args.respect_validity:
                gts_valid = np.isfinite(gts).all(axis=2)

        if gts is None or gts_valid is None:
            raise RuntimeError("Ground-truth data failed to load.")

        joint_names = None
        if args.skeleton is not None:
            try:
                skeleton = load_skeleton(args.skeleton, preds.shape[1])
                joint_names = [str(j.get("name", f"joint_{idx}")) for idx, j in enumerate(skeleton)]
            except Exception as exc:  # pragma: no cover - optional convenience
                print(f"Warning: failed to load skeleton {args.skeleton}: {exc}")

        label = _infer_label(result_path)
        metrics = compute_pck(
            preds,
            preds_valid,
            gts,
            gts_valid,
            args.pck_thresholds,
            args.exclude_frames,
            joint_names=joint_names,
        )

        _print_latex_table(label, metrics, [5.0, 10.0, 20.0])

        if args.save_json:
            if len(args.results) == 1:
                save_path = args.save_json
            else:
                save_path = args.save_json
                if save_path.suffix.lower() == ".json":
                    save_path = save_path.with_name(f"{save_path.stem}_{label}.json")
                else:
                    save_path = save_path / f"{label}.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
            print(f"\nMetrics written to {save_path}")

        if args.plot:
            if plt is None:
                print("matplotlib not installed; cannot plot.")
            elif not _ensure_interactive_backend():
                pass
            else:
                thresholds = metrics.get("thresholds", [])
                overall = metrics.get("overall", {})
                per_joint = metrics.get("per_joint", [])
                xs: List[float] = []
                ys: List[float] = []
                for t in sorted(thresholds):
                    key = f"pck@{t:g}"
                    val = overall.get(key)
                    if val is None:
                        continue
                    xs.append(t)
                    ys.append(val)
                if not xs:
                    print("No PCK data to plot.")
                else:
                    any_plots = True
                    plt.figure()
                    plt.plot(xs, ys, marker="o", linewidth=3.0, label="overall PCK", color="black")
                    cmap = plt.get_cmap("tab20")
                    colors = list(cmap.colors) if hasattr(cmap, "colors") else [
                        cmap(i) for i in np.linspace(0, 1, 20)
                    ]
                    linestyles = ["-", "--", "-.", ":"]
                    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "+", "x"]
                    for idx, entry in enumerate(per_joint):
                        local_x: List[float] = []
                        local_y: List[float] = []
                        for t in sorted(thresholds):
                            key = f"pck@{t:g}"
                            val = entry.get(key)
                            if val is None:
                                continue
                            local_x.append(t)
                            local_y.append(val)
                        if not local_x:
                            continue
                        color = colors[idx % len(colors)]
                        linestyle = linestyles[(idx // len(colors)) % len(linestyles)]
                        marker = markers[idx % len(markers)]
                        label_str = entry.get("label") or f"joint {entry['joint']}"
                        plt.plot(
                            local_x,
                            local_y,
                            marker=marker,
                            linewidth=1.2,
                            markersize=4,
                            alpha=0.65,
                            color=color,
                            linestyle=linestyle,
                            label=label_str,
                        )
                    plt.xlabel("Threshold mm")
                    plt.ylabel("PCK")
                    plt.ylim(0, 1.05)
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.title(f"PCK Evaluation: {label}")
                    plt.legend(
                        ncol=2,
                        fontsize="small",
                        framealpha=0.8,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                    )
                    plt.tight_layout()

    if args.plot and any_plots and plt is not None:
        plt.show()


if __name__ == "__main__":
    main()
