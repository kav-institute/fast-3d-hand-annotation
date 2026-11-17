#!/usr/bin/env python3
"""
Generate the workplace 3D PCK ablation table from triangulation outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from evaluate_pck import compute_pck, load_positions


EXCLUDE_FRAMES = [
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    78,
    79,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    166,
    182,
    183,
    184,
    185,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
]

METHODS = [
    ("Baseline", "baseline"),
    ("EWC", "candidates_grid"),
    ("AKR", "aggregated"),
    ("ABA", "aggregated_ba"),
    ("REM", "soft_marginal"),
]

SOURCES = [
    ("default", "Ours"),
    ("sapiens", "Sap."),
    ("mediapipe", "Med."),
]

TABLE_THRESHOLDS = [5.0, 10.0, 20.0]


@dataclass(frozen=True)
class PckMetrics:
    pck5: float
    pck10: float
    pck20: float


@dataclass(frozen=True)
class ResultMetrics:
    overall: PckMetrics
    per_joint: Dict[int, PckMetrics]


def _metric_key(threshold: float) -> str:
    return f"pck@{threshold:g}"


def _result_path(scene_root: Path, source: str, method: str) -> Path:
    return scene_root / f"outputs_{source}_{method}" / "triangulation_pipeline_outputs.npz"


def _metrics_from_entry(entry: Dict[str, object]) -> PckMetrics:
    return PckMetrics(
        pck5=float(entry[_metric_key(5.0)]),
        pck10=float(entry[_metric_key(10.0)]),
        pck20=float(entry[_metric_key(20.0)]),
    )


def _compute_result_metrics(
    result_path: Path,
    gt_positions,
    gt_validity,
    key: str,
    thresholds: Sequence[float],
    exclude_frames: Sequence[int],
    joint_indices: Sequence[int],
    respect_validity: bool,
) -> ResultMetrics:
    import numpy as np

    preds, preds_valid = load_positions(result_path, key)
    if not respect_validity:
        preds_valid = np.isfinite(preds).all(axis=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered.*")
        metrics = compute_pck(
            preds=preds,
            preds_valid=preds_valid,
            gts=gt_positions,
            gts_valid=gt_validity,
            thresholds=thresholds,
            exclude_frames=exclude_frames,
        )
    per_joint_entries = metrics["per_joint"]
    per_joint = {
        joint_idx: _metrics_from_entry(per_joint_entries[joint_idx])
        for joint_idx in joint_indices
    }
    return ResultMetrics(
        overall=_metrics_from_entry(metrics["overall"]),
        per_joint=per_joint,
    )


def _format_value(value: float, bold: bool) -> str:
    formatted = f"{value:.3f}"
    return f"\\textbf{{{formatted}}}" if bold else formatted


def _section_best(
    results: Dict[tuple[str, str], ResultMetrics],
    section: str,
    source: str,
    attr: str,
    joint_idx: Optional[int],
) -> float:
    values: List[float] = []
    for _, method in METHODS:
        result = results[(method, source)]
        metric = result.overall if section == "overall" else result.per_joint[int(joint_idx)]
        values.append(getattr(metric, attr))
    return max(values)


def _format_row(
    row_label: str,
    method: str,
    results: Dict[tuple[str, str], ResultMetrics],
    section: str,
    joint_idx: Optional[int],
    best: Dict[tuple[str, str, str], float],
    bold_scope: str,
) -> str:
    values: List[str] = []
    for attr in ("pck5", "pck10", "pck20"):
        for source, _ in SOURCES:
            result = results[(method, source)]
            metric = result.overall if section == "overall" else result.per_joint[int(joint_idx)]
            value = getattr(metric, attr)
            source_is_bolded = bold_scope == "all" or (bold_scope == "ours" and source == "default")
            is_best = source_is_bolded and abs(value - best[(section, source, attr)]) < 1e-12
            values.append(_format_value(value, is_best))
    return f"{row_label:<9} & " + " & ".join(values) + " \\\\"


def _append_section(
    lines: List[str],
    results: Dict[tuple[str, str], ResultMetrics],
    section: str,
    joint_idx: Optional[int],
    heading: Optional[str],
    bold_scope: str,
) -> None:
    if heading is not None:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{heading}}}}} \\\\")

    best = {
        (section, source, attr): _section_best(results, section, source, attr, joint_idx)
        for source, _ in SOURCES
        for attr in ("pck5", "pck10", "pck20")
    }
    for row_label, method in METHODS:
        lines.append(_format_row(row_label, method, results, section, joint_idx, best, bold_scope))


def _format_latex_table(
    results: Dict[tuple[str, str], ResultMetrics],
    best_joint_index: int,
    worst_joint_index: int,
    best_joint_label: str,
    worst_joint_label: str,
    caption: str,
    label: str,
    bold_scope: str,
) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "{\\setlength{\\tabcolsep}{3.2pt}%",
        "\\renewcommand{\\arraystretch}{0.95}",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{PCK$_{3D}$@5mm} & "
        "\\multicolumn{3}{c}{PCK$_{3D}$@10mm} & "
        "\\multicolumn{3}{c}{PCK$_{3D}$@20mm} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        " & Ours & Sap. & Med. & Ours & Sap. & Med. & Ours & Sap. & Med. \\\\",
        "\\midrule",
    ]
    _append_section(lines, results, "overall", None, None, bold_scope)
    _append_section(
        lines,
        results,
        "joint",
        best_joint_index,
        f"Best Joint: {best_joint_label}",
        bold_scope,
    )
    _append_section(
        lines,
        results,
        "joint",
        worst_joint_index,
        f"Worst Joint: {worst_joint_label}",
        bold_scope,
    )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_json(
    results: Dict[tuple[str, str], ResultMetrics],
    best_joint_index: int,
    worst_joint_index: int,
) -> str:
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for row_label, method in METHODS:
        out[row_label] = {}
        for source, source_label in SOURCES:
            result = results[(method, source)]
            out[row_label][source_label] = {
                "overall": {
                    "PCK@5mm": result.overall.pck5,
                    "PCK@10mm": result.overall.pck10,
                    "PCK@20mm": result.overall.pck20,
                },
                f"joint_{best_joint_index}": {
                    "PCK@5mm": result.per_joint[best_joint_index].pck5,
                    "PCK@10mm": result.per_joint[best_joint_index].pck10,
                    "PCK@20mm": result.per_joint[best_joint_index].pck20,
                },
                f"joint_{worst_joint_index}": {
                    "PCK@5mm": result.per_joint[worst_joint_index].pck5,
                    "PCK@10mm": result.per_joint[worst_joint_index].pck10,
                    "PCK@20mm": result.per_joint[worst_joint_index].pck20,
                },
            }
    return json.dumps(out, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("./data/2025_11_13_144123_pp_duplo_interation_p001_16fps_s001_7a5488a5"),
        help="Scene root containing outputs_<source>_<method> folders.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground-truth triangulation NPZ.",
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
        default="positions_graph",
        help="GT NPZ key.",
    )
    parser.add_argument(
        "--best-joint-index",
        type=int,
        default=29,
        help="Joint index for the best-joint section.",
    )
    parser.add_argument(
        "--best-joint-label",
        type=str,
        default="Index Tip Left",
        help="Printed label for --best-joint-index.",
    )
    parser.add_argument(
        "--worst-joint-index",
        type=int,
        default=20,
        help="Joint index for the worst-joint section.",
    )
    parser.add_argument(
        "--worst-joint-label",
        type=str,
        default="Pinky Tip Right",
        help="Printed label for --worst-joint-index.",
    )
    parser.add_argument(
        "--exclude-frames",
        type=int,
        nargs="*",
        default=EXCLUDE_FRAMES,
        help="Frame indices excluded from evaluation.",
    )
    parser.add_argument(
        "--respect-validity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use validity masks from NPZ inputs.",
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
        "--bold-scope",
        choices=["ours", "all", "none"],
        default="ours",
        help="Which best-per-column values to bold in the LaTeX output.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Ablation of 2D-to-3D lifting variants on the workplace evaluation dataset (PCK$_{3D}$). Higher is better.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:interhand_eval_own",
        help="LaTeX table label.",
    )
    args = parser.parse_args()
    if args.gt is None:
        args.gt = args.scene_root / 'outputs_labeled_gt/triangulation_pipeline_outputs.npz'
    return args


def main() -> None:
    args = parse_args()
    import numpy as np

    gt_positions, gt_validity = load_positions(args.gt, args.gt_key)
    if not args.respect_validity:
        gt_validity = np.isfinite(gt_positions).all(axis=2)

    joint_indices = [int(args.best_joint_index), int(args.worst_joint_index)]
    results: Dict[tuple[str, str], ResultMetrics] = {}
    for _, method in METHODS:
        for source, _ in SOURCES:
            result_path = _result_path(args.scene_root, source, method)
            if not result_path.exists():
                raise FileNotFoundError(f"Missing triangulation output: {result_path}")
            print(f"[run] {source} {method}", file=sys.stderr)
            results[(method, source)] = _compute_result_metrics(
                result_path=result_path,
                gt_positions=gt_positions,
                gt_validity=gt_validity,
                key=args.key,
                thresholds=TABLE_THRESHOLDS,
                exclude_frames=args.exclude_frames,
                joint_indices=joint_indices,
                respect_validity=bool(args.respect_validity),
            )

    if args.format == "json":
        text = _format_json(results, args.best_joint_index, args.worst_joint_index)
    else:
        text = _format_latex_table(
            results,
            best_joint_index=int(args.best_joint_index),
            worst_joint_index=int(args.worst_joint_index),
            best_joint_label=str(args.best_joint_label),
            worst_joint_label=str(args.worst_joint_label),
            caption=str(args.caption),
            label=str(args.label),
            bold_scope=str(args.bold_scope),
        )

    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
