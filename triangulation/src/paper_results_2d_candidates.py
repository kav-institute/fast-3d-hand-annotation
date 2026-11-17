#!/usr/bin/env python3
"""
Run the 2D candidate evaluator for the paper table and extract Overall hand metrics.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


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


@dataclass(frozen=True)
class DatasetConfig:
    label: str
    scene_root: Path
    gt_3d: Path
    gt_key: str | None = None
    hand_swap: str | None = None
    exclude_frames: Sequence[int] = ()
    skip_detectors: Sequence[str] = ()


@dataclass(frozen=True)
class HandMetrics:
    mean: float
    median: float
    pck5: float
    pck10: float
    pck20: float


@dataclass(frozen=True)
class RunResult:
    detector: str
    dataset: str
    command: List[str]
    metrics: Dict[str, HandMetrics]


METRIC_RE = re.compile(
    r"^(?P<name>oracle\(min\)|first_valid|top_score) hand-normalized: "
    r"NME_hand_mean=(?P<mean>[0-9.]+)% \| "
    r"NME_hand_median=(?P<median>[0-9.]+)% \| "
    r"PCK@5%hand=(?P<pck5>[0-9.]+) \| "
    r"PCK@10%hand=(?P<pck10>[0-9.]+) \| "
    r"PCK@20%hand=(?P<pck20>[0-9.]+)$"
)

DETECTOR_LABELS = {
    "mediapipe": "MediaPipe",
    "sapiens": "Sapiens2B",
    "interwild": "InterWild",
    "default": "Our",
}

METRIC_GROUPS = {
    "First Valid": "first_valid",
    "Best Candidate": "oracle(min)",
}


def _extract_overall_hand_metrics(output: str) -> Dict[str, HandMetrics]:
    in_overall = False
    metrics: Dict[str, HandMetrics] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == "Overall":
            in_overall = True
            continue
        if in_overall and line.startswith("Camera "):
            break
        if in_overall and " hand-normalized:" in line:
            match = METRIC_RE.match(line)
            if match is None:
                raise RuntimeError(f"Could not parse Overall hand-normalized line: {line}")
            metrics[match.group("name")] = HandMetrics(
                mean=float(match.group("mean")),
                median=float(match.group("median")),
                pck5=float(match.group("pck5")),
                pck10=float(match.group("pck10")),
                pck20=float(match.group("pck20")),
            )
    missing = {"oracle(min)", "first_valid", "top_score"} - set(metrics)
    if missing:
        raise RuntimeError(f"Missing Overall hand-normalized metrics: {sorted(missing)}")
    return metrics


def _run_eval(
    evaluator: Path,
    detector: str,
    dataset: DatasetConfig,
    extra_args: Sequence[str],
) -> RunResult:
    cmd = [
        sys.executable,
        str(evaluator),
        "--scene-root",
        str(dataset.scene_root),
        "--gt-3d",
        str(dataset.gt_3d),
        "--detection-source",
        detector,
    ]
    if dataset.gt_key is not None:
        cmd.extend(["--gt-key", dataset.gt_key])
    if dataset.hand_swap is not None:
        cmd.extend(["--hand-swap", dataset.hand_swap])
    if dataset.exclude_frames:
        cmd.append("--exclude-frames")
        cmd.extend(str(frame) for frame in dataset.exclude_frames)
    cmd.extend(extra_args)

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        details = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            f"Evaluation failed for detector={detector}, dataset={dataset.label}\n"
            f"Command: {' '.join(cmd)}\n{details}"
        )

    return RunResult(
        detector=detector,
        dataset=dataset.label,
        command=cmd,
        metrics=_extract_overall_hand_metrics(proc.stdout),
    )


def _format_metric_value(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def _format_table_row(detector: str, metrics: HandMetrics) -> str:
    return (
        f"{DETECTOR_LABELS[detector]:<9} & "
        f"{_format_metric_value(metrics.mean, 1)} & "
        f"{_format_metric_value(metrics.median, 1)} & "
        f"{_format_metric_value(metrics.pck5, 3)} & "
        f"{_format_metric_value(metrics.pck10, 3)} & "
        f"{_format_metric_value(metrics.pck20, 3)} \\\\"
    )


def _result_lookup(results: Sequence[RunResult]) -> Dict[tuple[str, str], RunResult]:
    return {(result.dataset, result.detector): result for result in results}


def _append_group(
    chunks: List[str],
    lookup: Dict[tuple[str, str], RunResult],
    dataset_label: str,
    group_label: str,
    detectors: Sequence[str],
) -> None:
    chunks.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{dataset_label} | {group_label}}}}} \\\\")
    metric_key = METRIC_GROUPS[group_label]
    for detector in detectors:
        result = lookup.get((dataset_label, detector))
        if result is None:
            continue
        chunks.append(_format_table_row(detector, result.metrics[metric_key]))


def _format_latex_table(results: Sequence[RunResult], datasets: Sequence[DatasetConfig]) -> str:
    lookup = _result_lookup(results)
    chunks = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation of raw 2D detection candidates. Mean and median are "
        "normalized by the projected GT hand bounding-box longest side and reported in percent.}",
        "\\label{tab:interhand_eval_2d_candidates_hand_norm}",
        "{%",
        "\\setlength{\\tabcolsep}{1.8pt}%",
        "\\renewcommand{\\arraystretch}{0.95}%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & NME$_{\\text{hand}}$ Mean & NME$_{\\text{hand}}$ Median & "
        "PCK$_{\\text{hand}}$@5 & PCK$_{\\text{hand}}$@10 & PCK$_{\\text{hand}}$@20 \\\\",
        "\\midrule",
    ]
    detector_order = ["mediapipe", "sapiens", "interwild", "default"]
    for dataset_idx, dataset in enumerate(datasets):
        dataset_detectors = [
            detector for detector in detector_order if detector not in dataset.skip_detectors
        ]
        _append_group(chunks, lookup, dataset.label, "First Valid", dataset_detectors)
        chunks.append("\\midrule")
        _append_group(chunks, lookup, dataset.label, "Best Candidate", dataset_detectors)
        if dataset_idx != len(datasets) - 1:
            chunks.append("\\midrule")
    chunks.extend(
        [
            "\\end{tabular}",
            "}",
            "\\begin{tabular}{lcc cc cc}",
            "\\toprule",
            "\\multicolumn{6}{@{}p{\\dimexpr\\linewidth-2\\tabcolsep\\relax}@{}}{%",
            "\\footnotesize MediaPipe results on InterHand2.6M scenes are omitted due to severe "
            "handedness and occlusion failure modes that make this comparison uninformative.}",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(chunks) + "\n"


def _format_command_comments(results: Sequence[RunResult]) -> str:
    chunks = ["% Commands used to generate this table:"]
    for result in results:
        chunks.append(f"% {result.detector} - {result.dataset}: {' '.join(result.command)}")
    return "\n".join(chunks) + "\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small-scene-root",
        type=Path,
        default=Path("./data/scene_2_light"),
        help="Scene root for the small-resolution dataset.",
    )
    parser.add_argument(
        "--small-gt-3d",
        type=Path,
        default=Path("./data/scene_2_light/gt_keypoints_3d.json"),
        help="3D GT for the small-resolution dataset.",
    )
    parser.add_argument(
        "--small-label",
        type=str,
        default="InterHand2.6M Scene 1",
        help="Label printed for the small-resolution dataset.",
    )
    parser.add_argument(
        "--second-small-scene-root",
        type=Path,
        default=Path("./data/scene_3_rom02_interaction_2_hand"),
        help="Scene root for the second small-resolution InterHand dataset.",
    )
    parser.add_argument(
        "--second-small-gt-3d",
        type=Path,
        default=Path(
            "./data/"
            "scene_3_rom02_interaction_2_hand/gt_keypoints_3d.json"
        ),
        help="3D GT for the second small-resolution InterHand dataset.",
    )
    parser.add_argument(
        "--second-small-label",
        type=str,
        default="InterHand2.6M Scene 2",
        help="Label printed for the second small-resolution InterHand dataset.",
    )
    parser.add_argument(
        "--large-scene-root",
        type=Path,
        default=Path(
            "./data/"
            "2025_11_13_144123_pp_duplo_interation_p001_16fps_s001_7a5488a5"
        ),
        help="Scene root for the large-resolution dataset.",
    )
    parser.add_argument(
        "--large-gt-3d",
        type=Path,
        default=Path(
            "./data/"
            "2025_11_13_144123_pp_duplo_interation_p001_16fps_s001_7a5488a5/"
            "outputs_labeled_gt/triangulation_pipeline_outputs.npz"
        ),
        help="3D GT for the large-resolution dataset.",
    )
    parser.add_argument(
        "--large-label",
        type=str,
        default="Own Dataset",
        help="Label printed for the large-resolution dataset.",
    )
    parser.add_argument(
        "--large-gt-key",
        type=str,
        default="positions_graph",
        help="GT key for the large NPZ dataset.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=["default", "sapiens", "mediapipe", "interwild"],
        default=["sapiens", "default", "mediapipe"],
        help="Detection sources to evaluate.",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).with_name("evaluate_2d_candidates.py"),
        help="Path to evaluate_2d_candidates.py.",
    )
    parser.add_argument(
        "--extra-eval-arg",
        action="append",
        default=[],
        help=(
            "Additional argument passed to every evaluator run. "
            "Repeat for multiple tokens, e.g. --extra-eval-arg=--top-k-candidates --extra-eval-arg=10."
        ),
    )
    parser.add_argument(
        "--include-commands",
        action="store_true",
        help="Include the exact evaluator commands as LaTeX comments before the table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional text file to write the extracted results to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [
        DatasetConfig(
            label=args.small_label,
            scene_root=args.small_scene_root,
            gt_3d=args.small_gt_3d,
            skip_detectors=("mediapipe",),
        ),
        DatasetConfig(
            label=args.second_small_label,
            scene_root=args.second_small_scene_root,
            gt_3d=args.second_small_gt_3d,
            skip_detectors=("mediapipe",),
        ),
        DatasetConfig(
            label=args.large_label,
            scene_root=args.large_scene_root,
            gt_3d=args.large_gt_3d,
            gt_key=args.large_gt_key,
            hand_swap="auto",
            exclude_frames=EXCLUDE_FRAMES,
        ),
    ]

    results: List[RunResult] = []
    for detector in args.detectors:
        for dataset in datasets:
            if detector in dataset.skip_detectors:
                print(f"[skip] detector={detector} dataset={dataset.label}", file=sys.stderr)
                continue
            print(f"[run] detector={detector} dataset={dataset.label}", file=sys.stderr)
            results.append(
                _run_eval(
                    evaluator=args.evaluator,
                    detector=detector,
                    dataset=dataset,
                    extra_args=args.extra_eval_arg,
                )
            )

    text = _format_latex_table(results, datasets)
    if args.include_commands:
        text = _format_command_comments(results) + text
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
