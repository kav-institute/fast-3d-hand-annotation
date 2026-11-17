#!/usr/bin/env python3
"""Visualize selected 2D joint detections across all cameras in a scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
    raise RuntimeError("Pillow is required for this visualization script.") from exc

from pipeline.dataloading import CameraModel, load_camera_parameters
from pipeline.geometry import project_point


DEFAULT_SCENE = "scene_3_rom02_interaction_2_hand"

SOURCE_STYLES = {
    "default": {"pattern": "detections_camera_{cam_id}_2d.npy", "color": (40, 220, 80)},
    "sapiens": {"pattern": "detections_sp_camera_{cam_id}_2d.npy", "color": (30, 170, 255)},
}
GT_COLOR = (255, 40, 40)
PRED_COLOR = (255, 215, 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path(f"./data/{DEFAULT_SCENE}"),
        help=f"{DEFAULT_SCENE} root directory.",
    )
    parser.add_argument("--frame", type=int, default=0, help="0-based frame index to visualize.")
    parser.add_argument(
        "--joint",
        type=int,
        nargs="+",
        required=True,
        help="Joint index or indices to draw, e.g. --joint 17.",
    )
    parser.add_argument(
        "--detection-source",
        choices=["default", "sapiens", "both"],
        default="default",
        help="Detector candidates to overlay.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maximum valid candidates per joint/source/camera to draw.",
    )
    parser.add_argument(
        "--label-top-k",
        type=int,
        default=3,
        help="Label this many top candidates per joint/source/camera.",
    )
    parser.add_argument(
        "--valid-threshold",
        type=float,
        default=0.5,
        help="Minimum detection valid flag.",
    )
    parser.add_argument(
        "--gt-3d",
        type=Path,
        default=Path(f"./data/{DEFAULT_SCENE}/gt_keypoints_3d.json"),
        help="Optional JSON 3D GT projected into each camera.",
    )
    parser.add_argument("--no-gt", action="store_true", help="Do not draw projected GT.")
    parser.add_argument(
        "--prediction",
        type=Path,
        default=None,
        help="Optional triangulation_pipeline_outputs.npz to project and draw.",
    )
    parser.add_argument(
        "--pred-key",
        type=str,
        default="positions_raw",
        help="Prediction NPZ key used with --prediction.",
    )
    parser.add_argument(
        "--camera-params",
        type=Path,
        default=None,
        help="Camera parameter JSON. Defaults to camera_parameter*.json in --scene-root.",
    )
    parser.add_argument("--cols", type=int, default=4, help="Number of camera tiles per row.")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale annotated camera tiles before composing the sheet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image. If omitted, the image is shown with OpenCV and not saved.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the composed image in an OpenCV window.",
    )
    return parser.parse_args()


def _camera_params_path(scene_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    matches = sorted(scene_root.glob("camera_parameter*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one camera_parameter*.json in {scene_root}, found {matches}")
    return matches[0]


def _source_names(source_arg: str) -> List[str]:
    if source_arg == "both":
        return ["default", "sapiens"]
    return [source_arg]


def _load_gt_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    keys = sorted(data.keys(), key=lambda k: (0, int(k)) if str(k).isdigit() else (1, k))
    positions: List[np.ndarray] = []
    valid: List[np.ndarray] = []
    for key in keys:
        entry = data[key]
        if isinstance(entry, dict) and "world_coord" in entry:
            frame = np.asarray(entry["world_coord"], dtype=np.float64)
            joint_valid = entry.get("joint_valid")
        else:
            frame = np.asarray(entry, dtype=np.float64)
            joint_valid = None
        positions.append(frame)
        if isinstance(entry, dict) and entry.get("hand_type_valid") is False:
            valid.append(np.zeros(frame.shape[0], dtype=bool))
        elif joint_valid is None:
            valid.append(np.isfinite(frame).all(axis=1))
        else:
            valid_arr = np.asarray(joint_valid, dtype=np.float64).reshape(-1)
            valid.append(valid_arr > 0.5)
    return np.stack(positions, axis=0), np.stack(valid, axis=0)


def _load_prediction(path: Path, key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")
    positions = np.asarray(data[key], dtype=np.float64)
    validity = data.get("validity")
    if validity is None or validity.shape[:2] != positions.shape[:2]:
        valid = np.isfinite(positions).all(axis=2)
    else:
        valid = np.asarray(validity, dtype=bool)
    return positions, valid


def _project_joint(
    positions: np.ndarray,
    valid: np.ndarray,
    frame: int,
    joint: int,
    camera: CameraModel,
) -> Optional[np.ndarray]:
    if frame < 0 or frame >= positions.shape[0] or joint < 0 or joint >= positions.shape[1]:
        return None
    if not valid[frame, joint] or not np.isfinite(positions[frame, joint]).all():
        return None
    X_h = np.array(
        [positions[frame, joint, 0], positions[frame, joint, 1], positions[frame, joint, 2], 1.0],
        dtype=np.float64,
    )
    projected = project_point(camera.P, X_h)
    if not np.isfinite(projected).all():
        return None
    return projected


def _camera_ids(scene_root: Path) -> List[str]:
    ids = []
    for camera_dir in sorted(scene_root.glob("camera_*")):
        if camera_dir.is_dir():
            ids.append(camera_dir.name.split("_", 1)[1])
    return ids


def _frame_image_path(scene_root: Path, cam_id: str, frame: int) -> Path:
    images = sorted((scene_root / f"camera_{cam_id}").glob("*.jpg"))
    if frame < 0 or frame >= len(images):
        raise IndexError(f"Frame {frame} is outside camera {cam_id} image range 0..{len(images) - 1}")
    return images[frame]


def _load_joint_candidates(
    det_path: Path,
    frame: int,
    joint: int,
    valid_threshold: float,
    top_k: int,
) -> np.ndarray:
    raw = np.load(det_path, mmap_mode="r")
    if raw.ndim != 5:
        raise ValueError(f"Unexpected detection tensor shape {raw.shape} in {det_path}")
    if frame < 0 or frame >= raw.shape[0]:
        raise IndexError(f"Frame {frame} is outside detection range 0..{raw.shape[0] - 1} in {det_path}")
    if joint < 0 or joint >= raw.shape[3]:
        raise IndexError(f"Joint {joint} is outside joint range 0..{raw.shape[3] - 1} in {det_path}")
    candidates = np.asarray(raw[frame, :, :, joint, :], dtype=np.float64).reshape(-1, 4)
    finite = np.isfinite(candidates[:, 0]) & np.isfinite(candidates[:, 1])
    valid = finite & (candidates[:, 3] > valid_threshold)
    candidates = candidates[valid]
    if candidates.size == 0:
        return candidates.reshape(0, 4)
    order = np.argsort(-candidates[:, 2])
    candidates = candidates[order]
    if top_k > 0:
        candidates = candidates[:top_k]
    return candidates


def _draw_cross(draw: ImageDraw.ImageDraw, xy: Sequence[float], color: Tuple[int, int, int], radius: int) -> None:
    x, y = float(xy[0]), float(xy[1])
    draw.line([(x - radius, y), (x + radius, y)], fill=color, width=3)
    draw.line([(x, y - radius), (x, y + radius)], fill=color, width=3)


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    xy: Sequence[float],
    color: Tuple[int, int, int],
    radius: int,
    fill: bool,
) -> None:
    x, y = float(xy[0]), float(xy[1])
    box = (x - radius, y - radius, x + radius, y + radius)
    if fill:
        draw.ellipse(box, fill=color, outline=color)
    else:
        draw.ellipse(box, outline=color, width=2)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Sequence[float],
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = float(xy[0]) + 6, float(xy[1]) - 12
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=color, font=font)


def _annotate_camera_tile(
    image_path: Path,
    cam_id: str,
    frame: int,
    joints: Sequence[int],
    sources: Sequence[str],
    scene_root: Path,
    cameras: Dict[str, CameraModel],
    gt: Optional[Tuple[np.ndarray, np.ndarray]],
    prediction: Optional[Tuple[np.ndarray, np.ndarray]],
    top_k: int,
    label_top_k: int,
    valid_threshold: float,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    header_h = 26
    tile = Image.new("RGB", (image.width, image.height + header_h), (18, 18, 18))
    tile.paste(image, (0, header_h))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    summary_parts: List[str] = []

    for joint in joints:
        for source in sources:
            style = SOURCE_STYLES[source]
            det_path = scene_root / str(style["pattern"]).format(cam_id=cam_id)
            if not det_path.exists():
                continue
            candidates = _load_joint_candidates(det_path, frame, joint, valid_threshold, top_k)
            summary_parts.append(f"{source[0]}j{joint}:{len(candidates)}")
            color = style["color"]
            for idx, cand in enumerate(candidates):
                xy = (float(cand[0]), float(cand[1]) + header_h)
                _draw_circle(draw, xy, color, radius=5 if idx == 0 else 3, fill=idx == 0)
                if idx < label_top_k:
                    _draw_label(draw, xy, f"{source[0]} j{joint} {cand[2]:.2f}", color, font)

        camera = cameras.get(cam_id)
        if camera is not None and gt is not None:
            projected = _project_joint(gt[0], gt[1], frame, joint, camera)
            if projected is not None:
                xy = (float(projected[0]), float(projected[1]) + header_h)
                _draw_cross(draw, xy, GT_COLOR, radius=8)
                _draw_label(draw, xy, f"GT j{joint}", GT_COLOR, font)
        if camera is not None and prediction is not None:
            projected = _project_joint(prediction[0], prediction[1], frame, joint, camera)
            if projected is not None:
                xy = (float(projected[0]), float(projected[1]) + header_h)
                _draw_circle(draw, xy, PRED_COLOR, radius=8, fill=False)
                _draw_label(draw, xy, f"P j{joint}", PRED_COLOR, font)

    title = f"cam {cam_id} | frame {frame} | {image_path.name}"
    if summary_parts:
        title = f"{title} | " + " ".join(summary_parts[:4])
    draw.text((6, 8), title, fill=(240, 240, 240), font=font)
    return tile


def _compose_grid(tiles: Sequence[Image.Image], cols: int, scale: float) -> Image.Image:
    if not tiles:
        raise ValueError("No camera tiles to compose.")
    if scale <= 0:
        raise ValueError("--scale must be positive.")
    scaled_tiles: List[Image.Image] = []
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for tile in tiles:
        if abs(scale - 1.0) < 1e-9:
            scaled_tiles.append(tile)
        else:
            w = max(1, int(round(tile.width * scale)))
            h = max(1, int(round(tile.height * scale)))
            scaled_tiles.append(tile.resize((w, h), resample=resample))

    cols = max(1, int(cols))
    rows = int(math.ceil(len(scaled_tiles) / cols))
    cell_w = max(tile.width for tile in scaled_tiles)
    cell_h = max(tile.height for tile in scaled_tiles)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    for idx, tile in enumerate(scaled_tiles):
        row, col = divmod(idx, cols)
        sheet.paste(tile, (col * cell_w, row * cell_h))
    return sheet


def _default_output(scene_root: Path, frame: int, joints: Sequence[int], source: str) -> Path:
    joint_text = "_".join(str(j) for j in joints)
    return scene_root / "visualizations" / f"joint_views_frame{frame:04d}_j{joint_text}_{source}.png"


def _show_with_opencv(image: Image.Image, window_name: str) -> None:
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("OpenCV is required for --show. Install opencv-python or pass --output.") from exc

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root
    sources = _source_names(args.detection_source)
    camera_params = _camera_params_path(scene_root, args.camera_params)
    cameras = load_camera_parameters(camera_params)

    gt = None
    if not args.no_gt and args.gt_3d.exists():
        gt = _load_gt_json(args.gt_3d)

    prediction = None
    if args.prediction is not None:
        prediction = _load_prediction(args.prediction, args.pred_key)

    tiles: List[Image.Image] = []
    for cam_id in _camera_ids(scene_root):
        image_path = _frame_image_path(scene_root, cam_id, args.frame)
        tiles.append(
            _annotate_camera_tile(
                image_path=image_path,
                cam_id=cam_id,
                frame=args.frame,
                joints=args.joint,
                sources=sources,
                scene_root=scene_root,
                cameras=cameras,
                gt=gt,
                prediction=prediction,
                top_k=args.top_k,
                label_top_k=args.label_top_k,
                valid_threshold=args.valid_threshold,
            )
        )

    sheet = _compose_grid(tiles, args.cols, args.scale)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.output)
        print(f"Wrote {args.output}")
    if args.show:
        joint_text = ",".join(str(joint) for joint in args.joint)
        _show_with_opencv(sheet, f"{DEFAULT_SCENE} frame {args.frame} joint {joint_text}")
    elif args.output is None:
        output = _default_output(scene_root, args.frame, args.joint, args.detection_source)
        print(f"No output written. Pass --output {output} or enable --show.")


if __name__ == "__main__":
    main()
