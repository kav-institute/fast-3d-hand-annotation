# Evaluation and reproduction notes

## Running the released workflows

The basic path is `detector/src/infer.py` → per-camera `.npy` candidates → `triangulation/src/run_multiview_triangulation.py` → `.npz` 3D results. All scripts expose `--help`.

The release is adapted from the two clean source repositories. Model architectures and core lifting algorithms are preserved; integration fixes are listed below. Reproducing the paper's numerical tables additionally requires the exact checkpoints, candidate tensors, capture/frame selection, calibration and ground truth. The released model and scene assets are available on [Zenodo: 10.5281/zenodo.24315081](https://doi.org/10.5281/zenodo.24315081); see the setup guide and its notes on numerical reproduction.

The [asset setup guide](interhand-setup.md) documents the bundled scene layout, InterHand camera/frame selections and optional image restoration. Scene 1 comes from the upstream train split and Scene 2 from the test split.

## Optional 2D sources

### Sapiens

Install detector dependencies and obtain the Sapiens-2B COCO WholeBody **TorchScript** checkpoint. A regular PyTorch state dict is not interchangeable with this file.

```bash
python detector/src/infer.py --source sapiens \
  --scene data/my_scene --checkpoint models/sapiens_wholebody.pt2 \
  --bboxes-json data/my_scene/bboxes.json --device cuda:0 --batch-size 1
```

MMDetection may be used instead of `--bboxes-json`, as for the learned model. The inherited Sapiens defaults are two scales (`1.0 1.5`) and nine offsets (`-9 0 9` along each axis). The learned-model defaults are four scales and 441 offsets. For a comparison using identical candidate-generation settings, explicitly specify the same `--scales` and `--offsets`. Do not assume the legacy defaults implement identical sampling budgets. Sapiens emits right-hand then left-hand joints from COCO WholeBody indices 112–132 and 91–111 respectively.

### MediaPipe

Use a separate environment to avoid installing both OpenCV wheels alongside each other:

```bash
python3 -m venv .venv-mediapipe
.venv-mediapipe/bin/python -m pip install -r requirements/mediapipe.txt
.venv-mediapipe/bin/python detector/src/infer.py --source mediapipe --scene data/my_scene
```

This adapter uses the legacy `solutions.hands` API. It applies 40 sequence-consistent scale/translation samples, with an unaugmented first sample. It preserves tracking within each camera/sample and records missing hands as invalid. Hand slots use automatic handedness; use `--images-mirrored` when the input is mirrored. This portable adapter omits the legacy random flips and corrects its BGR/RGB and validity handling. Fresh MediaPipe outputs therefore are **not a bit-for-bit regeneration of the original paper candidates**. The paper's manually corrected handedness assignments are not produced automatically by this adapter.

Use `--detection-source sapiens` or `mediapipe` in the lifting runner. The source selects file naming, not a different triangulation implementation.

## Evaluate one reconstruction

For millimeter-scale coordinates and JSON ground truth:

```bash
python triangulation/src/evaluate_pck_json.py \
  --results data/my_scene/outputs_default_aggregated_ba/triangulation_pipeline_outputs.npz \
  --gt data/my_scene/gt_keypoints_3d.json \
  --key positions_raw --pck-thresholds 5 10 20 --respect-validity
```

For NPZ ground truth, also set `--gt-key` to its coordinate array, e.g. `positions_graph`.

For 2D candidates:

```bash
python triangulation/src/evaluate_2d_candidates.py \
  --scene-root data/my_scene --gt-3d data/my_scene/gt_keypoints_3d.json \
  --detection-source default --hand-swap none
```

**Legacy metric conventions:** the 3D evaluators choose the better direct/swapped hand assignment using ground truth separately for each frame. They exclude wrists from the overall metric and compute PCK over finite evaluable predictions; missing predictions are not counted as failures in that denominator. Threshold comparison is `<=`. Some table scripts additionally exclude specific workplace frames. These scripts preserve the historical analysis conventions; their scores should not be interpreted as strict handedness-aware accuracy over every requested joint. Report evaluated counts/coverage and these conventions when using them on a new dataset. The 2D “Best Candidate” result is a ground-truth oracle, not a deployable candidate selector.

## Generate paper-style tables

The scripts write LaTeX by default (and expose additional options through `--help`). Run from the repository root. Create the output directory first:

```bash
mkdir -p outputs/tables
python triangulation/src/paper_results_2d_candidates.py \
  --include-commands --output outputs/tables/candidates.tex
python triangulation/src/paper_results_3d_interhand.py \
  --scene-root data/scene_3_rom02_interaction_2_hand \
  --output outputs/tables/interhand_interaction.tex
python triangulation/src/paper_results_3d_interhand.py \
  --scene-root data/scene_2_light \
  --output outputs/tables/interhand_light.tex
python triangulation/src/paper_results_3d_own.py \
  --scene-root data/2025_11_13_144123_pp_duplo_interation_p001_16fps_s001_7a5488a5 \
  --output outputs/tables/workplace.tex
python triangulation/src/paper_results_3d_own_refined.py \
  --scene-root data/2025_11_13_144123_pp_duplo_interation_p001_16fps_s001_7a5488a5 \
  --output outputs/tables/workplace_refined.tex
```

Use explicit scene names when recording results; the inherited scripts use different “Scene 1/2” label defaults, so confirm the intended mapping against the asset manifest before copying tables into a manuscript. The workplace scripts expect all configured methods/sources, including REM; the InterHand script omits REM. Existing archived outputs can be evaluated without rerunning the slow methods. `--gt` overrides ground truth; otherwise the 3D scripts derive its location from `--scene-root`.

The 2D table helper has separate root/GT arguments for each dataset. If using nondefault directories, supply both, e.g. `--small-scene-root ... --small-gt-3d ...`. Model checkpoints, assets and outputs are not inferred or downloaded.

## Changes from the clean source repositories

- Removed unused training/GPU/output settings, dead helpers, disabled experiments and the unused `h5py` dependency. The Re:InterHand loader now exposes only the outputs used by training; unused mask/MANO/NumPy-output options and discarded mesh-path lookups were removed. Original keypoint fits remain required.
- Separated `detector/` and `triangulation/` namespaces and dependency groups; scripts remain directly executable.
- Added portable model, data, output, device and sampling arguments. Inference no longer loads a training dataset or initializes models at import time, and it runs without interactive windows.
- Added bounding-box JSON input and filename alignment checks. Model creation does not download pretrained weights implicitly.
- Preserved the learned crop/scale sampling implementation and the Sapiens heatmap decoder. MMDetection now receives BGR arrays; Sapiens hand index names were clarified while retaining the canonical output order.
- Replaced the hard-coded training driver with a configurable loop using the original weighted dataset sampling, loss weights, optimizer and scheduler settings. Validation computes its own batch results; the old debug MPJPE code reused training tensors. Best checkpoints remain compatible with `model_state_dict` loading.
- Replaced the machine-specific RealSense demo with an optional webcam crop preview. This is not part of the multi-view pipeline.
- Validate calibration matrix consistency, duplicate IDs, detection dimensions and aligned frame counts instead of silently truncating camera streams. IDs containing underscores are supported.
- Graph refinement optimizes finite coordinates only, leaving joints with no observations missing instead of passing NaNs to SciPy. The objective for observed coordinates is unchanged.
- Legacy 3D evaluators handle entirely missing predictions without crashing. Their historical handedness/denominator conventions remain explicit above.

Training's original THAB unit conversion is configurable; defaults preserve the legacy convention. Full training reproducibility requires the original capture list, initialization, environments, sampling settings and data assets in addition to the source code. No claim of an independent full retraining is made by the smoke tests.
