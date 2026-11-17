"""
Skeleton handling and visualisation utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None


def load_skeleton(path: Path, joint_num: int) -> List[Dict[str, object]]:
    """
    Load skeleton definition (joint name + parent index per line).
    The format matches the provided hand skeleton file.
    """
    skeleton: List[Dict[str, object]] = [{} for _ in range(joint_num)]
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            joint_name, joint_id_str, parent_id_str = parts
            joint_id = int(joint_id_str)
            parent_id = int(parent_id_str)
            skeleton[joint_id]["name"] = joint_name
            skeleton[joint_id]["parent_id"] = parent_id
    for idx in range(len(skeleton)):
        children: List[int] = []
        for jdx in range(len(skeleton)):
            if skeleton[jdx].get("parent_id", -1) == idx:
                children.append(jdx)
        skeleton[idx]["child_id"] = children
    return skeleton


def get_keypoint_rgb(skeleton: Sequence[Dict[str, object]]) -> Dict[str, Tuple[int, int, int]]:
    """Assign a consistent RGB color per joint name."""
    rgb_dict: Dict[str, Tuple[int, int, int]] = {}
    for joint in skeleton:
        joint_name = str(joint.get("name", ""))
        if joint_name.endswith("thumb_null"):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith("thumb3"):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith("thumb2"):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith("thumb1"):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith("thumb0"):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith("index_null"):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith("index3"):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith("index2"):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith("index1"):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith("middle_null"):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith("middle3"):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith("middle2"):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith("middle1"):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith("ring_null"):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith("ring3"):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith("ring2"):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith("ring1"):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith("pinky_null"):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith("pinky3"):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith("pinky2"):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith("pinky1"):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
    return rgb_dict


def vis_keypoints_np(
    img: np.ndarray,
    kps: np.ndarray,
    skeleton: Sequence[Dict[str, object]],
    valid_mask: Optional[np.ndarray] = None,
    line_width: int = 3,
    circle_rad: int = 3,
) -> np.ndarray:
    """
    Draw 2D keypoints + skeleton on an RGB image using PIL.
    Returns a numpy array with the annotations.
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL (Pillow) is required for visualization but is not installed.")
    if valid_mask is None:
        valid_mask = np.isfinite(kps[:, 0]) & np.isfinite(kps[:, 1])
    rgb_dict = get_keypoint_rgb(skeleton)
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for joint_id, joint in enumerate(skeleton):
        parent_id = int(joint.get("parent_id", -1))
        joint_name = str(joint.get("name", joint_id))
        if not valid_mask[joint_id]:
            continue
        joint_xy = tuple(map(float, kps[joint_id]))
        color = rgb_dict.get(joint_name, (230, 230, 0))
        draw.ellipse(
            (
                joint_xy[0] - circle_rad,
                joint_xy[1] - circle_rad,
                joint_xy[0] + circle_rad,
                joint_xy[1] + circle_rad,
            ),
            fill=color,
        )
        if parent_id >= 0 and valid_mask[parent_id]:
            parent_name = str(skeleton[parent_id].get("name", parent_id))
            parent_xy = tuple(map(float, kps[parent_id]))
            parent_color = rgb_dict.get(parent_name, color)
            draw.line([joint_xy, parent_xy], fill=parent_color, width=line_width)
    return np.array(image, dtype=np.uint8)


def skeleton_edges(skeleton: Sequence[Dict[str, object]]) -> List[Tuple[int, int]]:
    """Return list of (parent, child) edges from the skeleton description."""
    edges: List[Tuple[int, int]] = []
    for joint_id, joint in enumerate(skeleton):
        parent_id = int(joint.get("parent_id", -1))
        if parent_id >= 0:
            edges.append((parent_id, joint_id))
    return edges
