#!/usr/bin/env python3
"""
Generate the InterHand2.6M 3D PCK ablation table from triangulation outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from evaluate_pck_json import compute_pck, load_positions_json, load_positions_npz


METHODS = [
    ("Baseline", "baseline"),
    ("EWC", "candidates_grid"),
    ("AKR", "aggregated"),
    ("ABA", "aggregated_ba"),
]

SOURCES = [
    ("default", "Ours"),
    ("sapiens", "Sap."),
]

TABLE_THRESHOLDS = [5.0, 10.0, 20.0]


@dataclass(frozen=True)
class PckMetrics:
    pck5: float
    pck10: float
    pck20: float


def _metric_key(threshold: float) -> str:
    return f"pck@{threshold:g}"


def _result_path(scene_root: Path, source: str, method: str) -> Path:
    return scene_root / f"outputs_{source}_{method}" / "triangulation_pipeline_outputs.npz"


def _load_gt_for_prediction(
    gt_path: Path,
    gt_key: str,
    frame_count: int,
    frame_start: Optional[int],
    respect_validity: bool,
) -> tuple[object, object]:
    if gt_path.suffix.lower() == ".json":
        gt_positions, gt_validity, _ = load_positions_json(
            gt_path,
            frame_start=frame_start,
            frame_count=frame_count,
        )
    else:
        gt_positions, gt_validity = load_positions_npz(gt_path, gt_key)

    if not respect_validity:
        import numpy as np

        gt_validity = np.isfinite(gt_positions).all(axis=2)
    return gt_positions, gt_validity


def _compute_result_metrics(
    result_path: Path,
    gt_path: Path,
    key: str,
    gt_key: str,
    thresholds: Sequence[float],
    frame_start: Optional[int],
    respect_validity: bool,
) -> PckMetrics:
    preds, preds_valid = load_positions_npz(result_path, key)
    if not respect_validity:
        import numpy as np

        preds_valid = np.isfinite(preds).all(axis=2)

    gt_positions, gt_validity = _load_gt_for_prediction(
        gt_path=gt_path,
        gt_key=gt_key,
        frame_count=preds.shape[0],
        frame_start=frame_start,
        respect_validity=respect_validity,
    )
    metrics = compute_pck(
        preds=preds,
        preds_valid=preds_valid,
        gts=gt_positions,
        gts_valid=gt_validity,
        thresholds=thresholds,
        exclude_frames=[],
    )
    overall = metrics["overall"]
    return PckMetrics(
        pck5=float(overall[_metric_key(5.0)]),
        pck10=float(overall[_metric_key(10.0)]),
        pck20=float(overall[_metric_key(20.0)]),
    )


def _format_value(value: float, bold: bool) -> str:
    formatted = f"{value:.3f}"
    return f"\\textbf{{{formatted}}}" if bold else formatted


def _best_by_column(
    results: Dict[tuple[str, str], PckMetrics],
    source: str,
    attr: str,
) -> float:
    values = [
        getattr(results[(method, source)], attr)
        for _, method in METHODS
        if (method, source) in results
    ]
    return max(values) if values else float("nan")


def _format_latex_table(results: Dict[tuple[str, str], PckMetrics], caption: str, label: str) -> str:
    best = {
        ("default", attr): _best_by_column(results, "default", attr)
        for attr in ("pck5", "pck10", "pck20")
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcc cc cc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{PCK$_{3D}$@5mm} & "
        "\\multicolumn{2}{c}{PCK$_{3D}$@10mm} & "
        "\\multicolumn{2}{c}{PCK$_{3D}$@20mm} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        " & Ours & Sap. & Ours & Sap. & Ours & Sap. \\\\",
        "\\midrule",
    ]

    for row_label, method in METHODS:
        row_values: List[str] = []
        for attr in ("pck5", "pck10", "pck20"):
            for source, _ in SOURCES:
                value = getattr(results[(method, source)], attr)
                is_best = source == "default" and abs(value - best[(source, attr)]) < 1e-12
                row_values.append(_format_value(value, is_best))
        lines.append(f"{row_label:<9} & " + " & ".join(row_values) + " \\\\")

    lines.extend(
        [
            "\\end{tabular}",
            "\\begin{tabular}{lcc cc cc}",
            "\\toprule",
            "\\multicolumn{7}{@{}p{\\dimexpr\\linewidth-2\\tabcolsep\\relax}@{}}{%",
            "\\footnotesize MediaPipe results are omitted on this dataset due to severe "
            "handedness/occlusion failure modes that make evaluation uninformative. "
            "REM is omitted due to long runtime.}",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_json(results: Dict[tuple[str, str], PckMetrics]) -> str:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row_label, method in METHODS:
        out[row_label] = {}
        for source, source_label in SOURCES:
            metrics = results[(method, source)]
            out[row_label][source_label] = {
                "PCK@5mm": metrics.pck5,
                "PCK@10mm": metrics.pck10,
                "PCK@20mm": metrics.pck20,
            }
    return json.dumps(out, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("./data/scene_3_rom02_interaction_2_hand"), # default=Path("./data/scene_2_light"), 3_rom02_interaction_2_hand
        help="Scene root containing outputs_default_* and outputs_sapiens_* folders.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,# Path("./data/scene_2_light/gt_keypoints_3d.json"),
        help="Ground-truth NPZ or JSON.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="positions_raw",
        help="Prediction NPZ key to evaluate.",
    )
    parser.add_argument(
        "--gt-key",
        type=str,
        default=None,
        help="GT NPZ key. Defaults to --key; ignored for JSON GT.",
    )
    parser.add_argument(
        "--gt-frame-start",
        type=int,
        default=None,
        help="Frame start index or frame id for JSON GT.",
    )
    parser.add_argument(
        "--respect-validity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use validity masks from NPZ/JSON inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the generated table.",
    )
    parser.add_argument(
        "--format",
        choices=["latex", "json"],
        default="latex",
        help="Output format.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Ablation of 2D-to-3D lifting variants on InterHand2.6M (PCK$_{3D}$). Higher is better.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:interhand_eval_interhand",
        help="LaTeX table label.",
    )
    args = parser.parse_args()
    if args.gt is None:
        args.gt = args.scene_root / 'gt_keypoints_3d.json'
    return args


def main() -> None:
    args = parse_args()
    gt_key = args.gt_key or args.key
    results: Dict[tuple[str, str], PckMetrics] = {}

    for _, method in METHODS:
        for source, _ in SOURCES:
            result_path = _result_path(args.scene_root, source, method)
            if not result_path.exists():
                raise FileNotFoundError(f"Missing triangulation output: {result_path}")
            print(f"[run] {source} {method}", file=sys.stderr)
            results[(method, source)] = _compute_result_metrics(
                result_path=result_path,
                gt_path=args.gt,
                key=args.key,
                gt_key=gt_key,
                thresholds=TABLE_THRESHOLDS,
                frame_start=args.gt_frame_start,
                respect_validity=bool(args.respect_validity),
            )

    if args.format == "json":
        text = _format_json(results)
    else:
        text = _format_latex_table(results, caption=args.caption, label=args.label)

    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
