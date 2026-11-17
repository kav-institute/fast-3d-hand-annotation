#!/usr/bin/env python3
"""
Pairwise PCK evaluation between two triangulation outputs.

Example:

    python evaluate_pck.py \
        --results scene_a/outputs/triangulation_pipeline_outputs.npz \
        --gt scene_b/outputs/triangulation_pipeline_outputs.npz \
        --key positions_refined \
        --pck-thresholds 5 10 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


def load_positions(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
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

    errors = errors[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]]
    all_errors = errors[np.isfinite(errors)]
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


def _print_latex_table(
    run_name: str,
    metrics: Dict[str, object],
    table_thresholds: Sequence[float],
) -> None:
    thresholds = [float(t) for t in table_thresholds]
    #overall = metrics.get("per_joint", {})[41]#
    #overall = metrics.get("per_joint", {})[8]# r index
    overall = metrics.get("overall", {})

    def format_val(val: Optional[float]) -> str:
        return "n/a" if val is None else f"{val:.3f}"

    mean_vals = [format_val(overall.get(f"pck@{t:g}")) for t in thresholds]
    print(f"{run_name} & " + " & ".join(mean_vals) + " \\\\")


###

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two triangulation outputs with PCK.")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        nargs="+",
        help="One or more predicted NPZs (triangulation_pipeline_outputs.npz).",
    )
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth NPZ (triangulation_pipeline_outputs.npz).")
    parser.add_argument("--key", default="positions_refined", help="NPZ key to compare (default: positions_refined).")
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

    gts, gts_valid = load_positions(args.gt, "positions_graph")  # args.key
    if not args.respect_validity:
        gts_valid = np.isfinite(gts).all(axis=2)

    any_plots = False
    for result_path in args.results:
        preds, preds_valid = load_positions(result_path, args.key)
        if not args.respect_validity:
            preds_valid = np.isfinite(preds).all(axis=2)

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
