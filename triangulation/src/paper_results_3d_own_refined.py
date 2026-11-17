#!/usr/bin/env python3
"""
Generate the workplace raw-vs-refined 3D PCK ablation table.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from evaluate_pck import load_positions
from paper_results_3d_own import (
    EXCLUDE_FRAMES,
    METHODS,
    SOURCES,
    TABLE_THRESHOLDS,
    PckMetrics,
    _compute_result_metrics,
    _format_value,
    _result_path,
)


SECTIONS = [
    ("Raw lifting without refinements", "positions_raw"),
    ("Structural and temporal refinements", "positions_refined"),
]


def _section_best(
    results: Dict[tuple[str, str, str], PckMetrics],
    section_key: str,
    source: str,
    attr: str,
) -> float:
    return max(
        getattr(results[(section_key, method, source)], attr)
        for _, method in METHODS
    )


def _format_row(
    row_label: str,
    method: str,
    section_key: str,
    results: Dict[tuple[str, str, str], PckMetrics],
    best: Dict[tuple[str, str, str], float],
    bold_scope: str,
) -> str:
    values: List[str] = []
    for attr in ("pck5", "pck10", "pck20"):
        for source, _ in SOURCES:
            value = getattr(results[(section_key, method, source)], attr)
            source_is_bolded = bold_scope == "all" or (bold_scope == "ours" and source == "default")
            is_best = source_is_bolded and abs(value - best[(section_key, source, attr)]) < 1e-12
            values.append(_format_value(value, is_best))
    return f"{row_label:<9} & " + " & ".join(values) + " \\\\"


def _format_tabular(results: Dict[tuple[str, str, str], PckMetrics], bold_scope: str) -> str:
    lines = [
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{PCK$_{3D}$@5mm} & "
        "\\multicolumn{3}{c}{PCK$_{3D}$@10mm} & "
        "\\multicolumn{3}{c}{PCK$_{3D}$@20mm} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        " & Ours & Sap. & Med. & Ours & Sap. & Med. & Ours & Sap. & Med. \\\\",
        "\\midrule",
    ]
    for section_idx, (section_label, section_key) in enumerate(SECTIONS):
        if section_idx > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{section_label}}}}} \\\\")
        best = {
            (section_key, source, attr): _section_best(results, section_key, source, attr)
            for source, _ in SOURCES
            for attr in ("pck5", "pck10", "pck20")
        }
        for row_label, method in METHODS:
            lines.append(_format_row(row_label, method, section_key, results, best, bold_scope))
    lines.extend(["", "\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _format_json(results: Dict[tuple[str, str, str], PckMetrics]) -> str:
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for section_label, section_key in SECTIONS:
        out[section_label] = {}
        for row_label, method in METHODS:
            out[section_label][row_label] = {}
            for source, source_label in SOURCES:
                metrics = results[(section_key, method, source)]
                out[section_label][row_label][source_label] = {
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
        default=Path("./data/2025_11_13_144123_pp_duplo_interation_p001_16fps_s001_7a5488a5"),
        help="Scene root containing outputs_<source>_<method> folders.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground-truth triangulation NPZ.",
    )
    parser.add_argument("--raw-key", default="positions_raw", help="Prediction key for raw lifting.")
    parser.add_argument(
        "--refined-key",
        default="positions_refined",
        help="Prediction key for structural/temporal refinements.",
    )
    parser.add_argument("--gt-key", default="positions_graph", help="GT NPZ key.")
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
        "--bold-scope",
        choices=["ours", "all", "none"],
        default="ours",
        help="Which best-per-column values to bold in the LaTeX output.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output file.")
    parser.add_argument("--format", choices=["latex", "json"], default="latex", help="Output format.")
    args = parser.parse_args()
    if args.gt is None:
        args.gt = args.scene_root / 'outputs_labeled_gt/triangulation_pipeline_outputs.npz'
    return args


def main() -> None:
    args = parse_args()
    import numpy as np

    section_keys = [
        ("Raw lifting without refinements", str(args.raw_key)),
        ("Structural and temporal refinements", str(args.refined_key)),
    ]
    global SECTIONS
    SECTIONS = section_keys

    gt_positions, gt_validity = load_positions(args.gt, args.gt_key)
    if not args.respect_validity:
        gt_validity = np.isfinite(gt_positions).all(axis=2)

    results: Dict[tuple[str, str, str], PckMetrics] = {}
    for _, section_key in SECTIONS:
        for _, method in METHODS:
            for source, _ in SOURCES:
                result_path = _result_path(args.scene_root, source, method)
                if not result_path.exists():
                    raise FileNotFoundError(f"Missing triangulation output: {result_path}")
                print(f"[run] key={section_key} {source} {method}", file=sys.stderr)
                with contextlib.redirect_stdout(io.StringIO()):
                    metrics = _compute_result_metrics(
                        result_path=result_path,
                        gt_positions=gt_positions,
                        gt_validity=gt_validity,
                        key=section_key,
                        thresholds=TABLE_THRESHOLDS,
                        exclude_frames=args.exclude_frames,
                        joint_indices=[],
                        respect_validity=bool(args.respect_validity),
                    )
                results[(section_key, method, source)] = metrics.overall

    text = _format_json(results) if args.format == "json" else _format_tabular(results, args.bold_scope)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
