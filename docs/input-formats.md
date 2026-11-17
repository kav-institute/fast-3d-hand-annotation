# Input and output formats

## Scene and image alignment

Use exactly one `camera_parameter*.json` file per scene. Camera IDs are strings: `"0"`, `"400042"`, or `"front_left"` are valid. Image folders are `camera_<id>` and detection file IDs must match calibration IDs.

Images must already be synchronized. The detector checks identical file names across cameras, then sorts them lexicographically. Use names such as `000000.jpg`, `000001.jpg`. Images within one camera must have the same dimensions. `--image-extension png` selects PNG files. Candidate-only lifting cannot check timestamps: row `f` must represent the same instant in every tensor. Differing frame/joint counts are rejected.

## Calibration JSON

The root object has a `cameras` array. Each entry contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `camera` | string or integer | Unique camera ID |
| `camera_matrix` | 3 × 3 | Intrinsic matrix K, in pixels |
| `distortion_coeff` | vector | OpenCV pinhole distortion coefficients, or empty for none |
| `rotation_matrix` | 3 × 3 | World-to-camera rotation R |
| `t_param` | 3 or 3 × 1 | World-to-camera translation t |
| `projection_matrix` | 3 × 4 | K [R \| t], up to a common nonzero scale |

The convention is `X_camera = R @ X_world + t`; the camera center in world coordinates is `-R.T @ t`. `projection_matrix` is the undistorted pinhole projection. Candidate points are undistorted internally before triangulation. Do not feed undistorted images with the old nonzero distortion calibration. Fisheye calibration must be converted/rectified to the supported pinhole model first.

Example entry for a camera at world position `(100, 0, 0)`:

```json
{
  "cameras": [{
    "camera": "0",
    "camera_matrix": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
    "distortion_coeff": [0, 0, 0, 0, 0],
    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "t_param": [[-100], [0], [0]],
    "projection_matrix": [[800, 0, 320, -80000], [0, 800, 240, 0], [0, 0, 1, 0]]
  }]
}
```

Add at least one other calibrated camera for reconstruction. Baselines must have adequate separation for triangulation. All world-space coordinates and translations must share units. Lifting preserves those units; it does not convert centimeters/meters to millimeters. The example generator uses millimeters.

## Candidate tensors

Save a numeric NumPy `.npy` array of shape `[F, S, O, 42, 4]`:

- `F`: synchronized frames.
- `S`: crop scales (or 1 for arbitrary candidate sets).
- `O`: crop offsets/samples. It can contain flattened arbitrary hypotheses.
- `42`: joints, right hand then left hand.
- Four channels: **x, y, score, valid**. Coordinates are in the original distorted image's pixel frame. A validity value greater than 0.5 means usable. Missing detections have validity 0. Valid detections must have finite coordinates and scores.

Naming:

```text
detections_camera_<id>_2d.npy      --detection-source default
detections_sp_camera_<id>_2d.npy   --detection-source sapiens
detections_mp_camera_<id>_2d.npy   --detection-source mediapipe
```

`S` and `O` are flattened into the candidate dimension. Use `[F,1,N,42,4]` for another detector. Baseline takes the first candidate, so candidate ordering matters. The inherited ANT/EWC/AKR configurations retain the first 100 candidates, whereas REM uses all candidates. These are code defaults, not a guarantee that every emitted sample contributes to lifting.

Each hand uses MediaPipe's 21-point order: wrist; thumb CMC/MCP/IP/tip (indices 1–4); then index, middle, ring and pinky MCP/PIP/DIP/tip chains. Right wrist is 0, left wrist 21. Do not assume a dataset's native InterHand order matches this layout; use the supplied conversion helper with the correct skeleton files.

## Optional bounding boxes

`--bboxes-json` bypasses the MMDetection stage. The file maps camera IDs to boxes in exactly the same sorted frame order as the images:

```json
{
  "0": [[100, 80, 420, 340], null],
  "1": [[120, 75, 450, 350], [125, 80, 455, 355]]
}
```

Boxes are `[left, top, right, bottom]` pixel coordinates enclosing both hands; `null` means no box detected. The inherited temporal box filling/smoothing is applied before crop sampling. A camera with no valid boxes produces invalid candidates. Ground-truth keypoints are not needed for inference.

## 3D ground truth and output

JSON ground truth is a frame-keyed object:

```json
{
  "0": {"world_coord": [[0, 0, 800], [1, 2, 801]], "joint_valid": [1, 1]}
}
```

The example above abbreviates the joints; actual hand evaluation requires all 42 entries in the canonical order. Frames are numerically sorted when possible. `joint_valid` is optional; `hand_type_valid: false` invalidates a frame. Coordinates must be in the same world frame and units as the prediction. NPZ ground truth with a selected `positions_*` key and optional `validity` mask is also accepted.

The runner writes:

- `triangulation_pipeline_outputs.npz`: `positions_raw [F,42,3]`, `validity [F,42]`, `residual_sigmas [F,42]`; optional refinement adds `positions_interpolated`, `positions_refined`, `positions_graph`, and `manual_annotations`.
- `triangulation_label_stage.npz`: raw positions, raw validity, residuals, and an initially false manual-annotation mask.

The uncertainty array is a solver-specific reprojection residual summary; do not interpret it as a calibrated 3D error in millimeters. Refinement may fill gaps, so its validity mask differs from raw validity. A joint absent for the whole sequence remains NaN.

## THAB training annotations

The THAB loader expects:

```text
thab_training/
  skeleton_thab.txt
  annotations__train.json
  annotations__val.json
  sequence_name/
    camera_parameters.json
    camera_0/000000.jpg
    ...
```

COCO-style JSON has `images` and `annotations` arrays. Image entries need `id`, `file_name` relative to the dataset root, and `camera` (e.g. `camera_0`). Annotation entries need `id`, `image_id`, `keypoints` as a list of `[x,y]` pairs (not flattened COCO `[x,y,v]` triplets), `keypoints_3d` as world `[x,y,z]` triples, and `hand_type` equal to `right`, `left`, or `interacting`. Single-hand entries may contain 21 points; two-hand entries contain 42.

The legacy THAB data convention is centimeters for `keypoints_3d` and calibration translations. The training loader converts both to millimeters using `--thab-unit-scale` (default 10; use 1 for millimeter inputs). This option affects THAB training only, not triangulation or Re:InterHand.

`skeleton_thab.txt` has 42 lines of `joint_name joint_id parent_id`, with -1 for each wrist root. It must describe the canonical order. Use the dataset's supplied skeleton; conversion between dataset orders must preserve joint names and handedness.

The optional Re:InterHand loader consumes its original capture directories, `frame_list.txt`, camera parameters, images, and original 3D fits. Obtain Re:InterHand separately and pass `--reinterhand-root` and explicit `--capture-ids`. Its loader uses the THAB skeleton to map joint names. This source-data workflow is separate from the selected InterHand2.6M evaluation scenes (see [their split provenance](interhand-setup.md#exact-interhand-subsets)).
