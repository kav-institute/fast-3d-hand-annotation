#!/usr/bin/env python3
"""Reorder JSON 3D GT keypoints from one named skeleton layout to another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence


def load_skeleton_names(path: Path) -> Dict[int, str]:
    """Load skeleton files in the local ``name index parent`` format."""
    names: Dict[int, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"{path}:{line_no}: expected 'name index parent'")

        name, idx_text = parts[0], parts[1]
        try:
            idx = int(idx_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid joint index {idx_text!r}") from exc

        if idx in names:
            raise ValueError(f"{path}:{line_no}: duplicate joint index {idx}")
        names[idx] = name

    if not names:
        raise ValueError(f"{path}: no skeleton joints found")
    return names


def make_permutation(source_names: Dict[int, str], target_names: Dict[int, str]) -> List[int]:
    source_by_name = {name: idx for idx, name in source_names.items()}
    if len(source_by_name) != len(source_names):
        raise ValueError("source skeleton contains duplicate joint names")

    target_indices = sorted(target_names)
    if target_indices != list(range(len(target_indices))):
        raise ValueError("target skeleton indices must be contiguous from 0")

    missing = [target_names[idx] for idx in target_indices if target_names[idx] not in source_by_name]
    if missing:
        raise ValueError(f"source skeleton is missing target joints: {missing}")

    return [source_by_name[target_names[idx]] for idx in target_indices]


def reorder_sequence(values: Sequence[object], permutation: Sequence[int], field_name: str) -> List[object]:
    if len(values) != len(permutation):
        raise ValueError(
            f"{field_name} has {len(values)} joints, expected {len(permutation)}"
        )
    return [values[source_idx] for source_idx in permutation]


def reorder_gt_json(data: object, permutation: Sequence[int]) -> int:
    """Reorder ``world_coord`` and optional ``joint_valid`` entries in place."""
    if isinstance(data, MutableMapping):
        entries = data.values()
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("GT JSON root must be a dict or list")

    changed = 0
    for entry in entries:
        if not isinstance(entry, MutableMapping) or "world_coord" not in entry:
            continue
        entry["world_coord"] = reorder_sequence(
            entry["world_coord"], permutation, "world_coord"
        )
        if "joint_valid" in entry and entry["joint_valid"] is not None:
            entry["joint_valid"] = reorder_sequence(
                entry["joint_valid"], permutation, "joint_valid"
            )
        changed += 1
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert 3D keypoint GT JSON from a source skeleton order to a target "
            "skeleton order by matching joint names."
        )
    )
    parser.add_argument("--gt", type=Path, required=True, help="Input gt_keypoints_3d.json")
    parser.add_argument(
        "--source-skeleton",
        type=Path,
        required=True,
        help="Skeleton layout currently used by the GT JSON, e.g. InterHand skeleton.txt",
    )
    parser.add_argument(
        "--target-skeleton",
        type=Path,
        default=Path("data/scene_2_light/hand_skeleton.txt"),
        help="Skeleton layout to write. Defaults to the repo/MediaPipe-like scene skeleton.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Omit with --in-place to overwrite the input file.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --gt. Use only once per raw-source-layout GT file.",
    )
    parser.add_argument(
        "--print-permutation",
        action="store_true",
        help="Print the target-index to source-index permutation before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None and not args.in_place:
        raise SystemExit("Pass --output or --in-place")
    if args.output is not None and args.in_place:
        raise SystemExit("Use either --output or --in-place, not both")

    source_names = load_skeleton_names(args.source_skeleton)
    target_names = load_skeleton_names(args.target_skeleton)
    permutation = make_permutation(source_names, target_names)
    if args.print_permutation:
        print(permutation)

    data = json.loads(args.gt.read_text(encoding="utf-8"))
    changed = reorder_gt_json(data, permutation)

    output = args.gt if args.in_place else args.output
    assert output is not None
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Reordered {changed} frame entries: {args.gt} -> {output}")


if __name__ == "__main__":
    main()
