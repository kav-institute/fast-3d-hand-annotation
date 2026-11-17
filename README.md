# Fast 3D Hand Annotation

Code for **Fast 3D Hand Annotation via Multi-View Pseudo-Labels and Aggregated Nonlinear Triangulation**, by Andreas Hubert, Konrad Doll, and Bernhard Sick. Accepted at **ACDSA 2027**.

The pipeline generates 2D hand-keypoint candidates from synchronized camera images, robustly aggregates the candidates, and reconstructs 3D pseudo-labels using calibrated multi-view geometry. It includes the HRNet-W48/SimCC detector, training and inference tools, ANT and four comparison methods, optional temporal/anatomical refinement, and evaluation scripts.

**Models and data:** the asset bundle is available on [Zenodo: 10.5281/zenodo.24315081](https://doi.org/10.5281/zenodo.24315081), including our trained hand-keypoint model, scene data and checksums. There are no model weights or research recordings in this repository. The third-party hand detector and Sapiens model can already be obtained using the [download instructions below](#download-pretrained-models). The synthetic example below runs without them. See [asset requirements](docs/assets.md).

## Repository layout

```text
detector/src/       Model, dataset loaders, training and 2D candidate generation
triangulation/src/ Multi-view lifting, refinement, evaluation and visualization
examples/          Synthetic scene generator (no external assets)
tests/             CPU geometry, model and integration tests
docs/              Input formats, assets, reproduction notes and test scope
requirements/      Optional detector and MediaPipe dependencies
```

Run the examples below from the repository root. In the command examples, replace scene and checkpoint paths with your actual files.

## Install

Python **3.12** on Linux is the tested environment. CPU triangulation does not require PyTorch or a GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the learned detector and training, install matching PyTorch/torchvision builds for your platform, then the detector dependencies. For CPU:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/detector.txt
```

For CUDA, select the appropriate PyTorch build for your driver/GPU instead of the CPU command. Inference accepts `--device cuda:0`; training defaults to that device. The code does not download backbone weights unless training is explicitly given `--pretrained-backbone`.

Bounding-box detection additionally needs **MMDetection 3.x and its compatible MMCV native operators**. It is optional when you supply `--bboxes-json`. This stack has separate PyTorch/CUDA constraints; follow its [installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html) in a compatible environment. Installing the detector requirements alone does not install MMDetection.

## Quick start: no models or research data needed

```bash
python examples/make_synthetic_scene.py --output data/synthetic
python triangulation/src/run_multiview_triangulation.py \
  --detection-source-path data/synthetic \
  --method aggregated_ba --detection-source default
```

The generator creates four calibrated cameras, 12 frames, 42 joints and nine candidates per joint/view. Its output directory must not already exist. Results are saved to:

```text
data/synthetic/outputs_default_aggregated_ba/triangulation_pipeline_outputs.npz
```

Inspect the error against the synthetic ground truth:

```bash
python - <<'PY'
import numpy as np
with np.load('data/synthetic/ground_truth.npz') as gt, np.load(
    'data/synthetic/outputs_default_aggregated_ba/triangulation_pipeline_outputs.npz'
) as result:
    error = np.linalg.norm(result['positions_raw'] - gt['positions_raw'], axis=-1)
    print('Valid joints:', result['validity'].sum())
    print('Maximum error [mm]:', np.nanmax(error))
PY
```

With the default noise-free inputs, the error should be below 0.001 mm. This verifies geometry and file interoperability; it is not a paper benchmark.

## Set up the released assets

Download the asset bundle from [Zenodo: 10.5281/zenodo.24315081](https://doi.org/10.5281/zenodo.24315081). Extracting `models.zip` and `scenes.zip` provides `models/hand_keypoints.pt` and three scene folders under `scenes/`. Follow the [asset setup and InterHand guide](docs/interhand-setup.md) to verify `SHA256SUMS` and place the files at the paths used below.

InterHand images are excluded from the bundle. The guide records the exact camera IDs, source sequences and ordered frame lists for both scenes, and includes a setup script to restore images from your own InterHand2.6M download. Images are optional when using the supplied candidates or saved reconstructions.

## Inputs

For candidate generation, prepare a scene like this:

```text
data/my_scene/
  camera_parameters.json
  camera_0/000000.jpg
  camera_0/000001.jpg
  camera_1/000000.jpg
  camera_1/000001.jpg
  ...
```

- Use at least two cameras. Folder IDs must match the calibration IDs.
- Images must be synchronized, with identical, lexicographically sortable filenames across cameras. Use zero-padded frame numbers. File names are checked; temporal synchronization itself is the caller's responsibility.
- Calibration contains intrinsics, distortion, and **world-to-camera** extrinsics, with `P = K [R | t]`. Coordinates and translations must use the same units. Use millimeters for the evaluation thresholds in the paper.
- The joint layout is **right hand 0–20, left hand 21–41**, each in MediaPipe order. Wrists are 0 and 21.
- Existing detections may be supplied directly; images and checkpoints are then unnecessary for triangulation.

The exact calibration, candidate tensor, bounding-box and training annotation formats are documented in [input formats](docs/input-formats.md).

## Download pretrained models

Run these commands from the repository root. Our learned hand-keypoint checkpoint (`models/hand_keypoints.pt`) is included in `models.zip` on [Zenodo: 10.5281/zenodo.24315081](https://doi.org/10.5281/zenodo.24315081). The two third-party models below are available upstream now.

### Hand bounding-box detector

Use the **Cascade R-CNN X-101-64x4d-FPN trained on OneHand10K**, matching the configuration included in this repository. Download the checkpoint (about 509 MB) from OpenMMLab:

```bash
mkdir -p models
curl --fail --location --retry 3 \
  'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth' \
  --output models/hand_detector.pth
```

The original scripts passed this URL to MMDetection's `init_detector`, which downloaded and cached the weights automatically. This release's `infer.py` expects a **local file** via `--detector-checkpoint models/hand_detector.pth`; it does not automatically download this checkpoint. You can skip the detector entirely by supplying `--bboxes-json`.

### Sapiens keypoint model (optional comparison)

Use **Sapiens-2B COCO WholeBody, 133 keypoints, TorchScript**, with the original filename `sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2` (about 8.67 GB). The [Sapiens model documentation](https://github.com/facebookresearch/sapiens/blob/main/lite/docs/POSE_README.md) links to the [Hugging Face download repository](https://huggingface.co/noahcao/sapiens-pose-coco). Log in on that page and accept the model's access conditions before downloading; access is currently gated.

Then install the Hugging Face CLI, authenticate with the same account, and download just the required file:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
mkdir -p models
SAPIENS_FILE='sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2'
hf download noahcao/sapiens-pose-coco "$SAPIENS_FILE" --local-dir models/upstream
mv -n "models/upstream/$SAPIENS_FILE" models/sapiens_wholebody.pt2
```

Alternatively, download that file from the repository's Files tab and save it as `models/sapiens_wholebody.pt2`. Select the **133-keypoint TorchScript** checkpoint: the 17-keypoint, 308-keypoint and other export variants are not interchangeable with this adapter. Follow the upstream model terms linked on the download page.

Generate Sapiens candidates using the same hand bounding-box detector:

```bash
python detector/src/infer.py \
  --source sapiens \
  --scene data/my_scene \
  --checkpoint models/sapiens_wholebody.pt2 \
  --detector-checkpoint models/hand_detector.pth \
  --device cuda:0
```

## Generate 2D candidates

Once the assets are available, generate candidates with the learned model and a local MMDetection checkpoint:

```bash
python detector/src/infer.py \
  --scene data/my_scene \
  --checkpoint models/hand_keypoints.pt \
  --detector-checkpoint models/hand_detector.pth \
  --device cuda:0
```

Alternatively, bypass MMDetection using per-frame bounding boxes:

```bash
python detector/src/infer.py \
  --scene data/my_scene \
  --checkpoint models/hand_keypoints.pt \
  --bboxes-json data/my_scene/bboxes.json \
  --device cpu --scales 1.3 --offsets 0 --batch-size 1
```

The second command uses one crop per frame as a cheap setup check. For the original learned-model sampling defaults, omit `--scales` and `--offsets`: four scales × 21 × 21 pixel offsets = **1,764 candidates per joint/view**. This can be expensive; `--batch-size` controls peak inference batch memory and `--max-frames` can limit a smoke test.

Outputs are `detections_camera_<id>_2d.npy`, shaped `[frames, scales, offsets, 42, 4]`, with `(x_px, y_px, score, valid)` in original distorted-image coordinates. `detections_manifest.json` records CLI arguments and processed image filenames. The learned source currently assigns unit scores to finite predictions; these are not calibrated confidences. Inference runs without a display and refuses to replace existing detections unless given `--overwrite`.

Sapiens and MediaPipe adapters, their sampling conventions and dependencies are described in [reproduction notes](docs/reproduction.md). The original `reconstruct_eval*.py` names are retained as wrappers around `infer.py`.

## Triangulation and refinement

| Paper name | CLI method |
| --- | --- |
| Single Candidate RANSAC | `baseline` |
| Epipolar-Weighted Candidates (EWC) | `candidates_grid` |
| Aggregated Keypoint RANSAC (AKR) | `aggregated` |
| Aggregated Nonlinear Triangulation (ANT) | `aggregated_ba` |
| RANSAC with Expectation-Maximization (REM) | `soft_marginal` |

```bash
python triangulation/src/run_multiview_triangulation.py \
  --detection-source-path data/my_scene \
  --method aggregated_ba --detection-source default --run-post
```

Use `--method baseline candidates_grid aggregated aggregated_ba` for a comparison, or add `soft_marginal` for REM. RANSAC-based methods, especially REM, can be very slow at the default 5,000 iterations. Tune `--ransac-workers` and `--epi-grid-workers` for your machine (both default to 32). ANT does not run RANSAC.

Each method/source gets an `outputs_<source>_<method>/` directory. The main NPZ contains `positions_raw [F,42,3]`, `validity [F,42]` and `residual_sigmas [F,42]`. Invalid raw coordinates are NaN. With `--run-post`, it also contains `positions_interpolated`, `positions_refined` and `positions_graph`. Refinement includes a potentially expensive SciPy graph optimization; `--post-graph-window` controls its window size. The label-stage NPZ preserves the raw validity mask; the postprocessed NPZ's `validity` includes interpolated joints. See [reproduction notes](docs/reproduction.md) for evaluation conventions.

## Train the detector

Training needs THAB-format training/validation annotations and images; the evaluation scenes alone are insufficient. Use `--thab-unit-scale 1` for annotations and translations in millimeters, or `10` for the legacy THAB centimeter convention (the default).

```bash
python detector/src/train.py \
  --dataset-root data/thab_training \
  --thab-unit-scale 10 \
  --reinterhand-root data/reinterhand \
  --capture-ids CAPTURE_DIRECTORY_1 CAPTURE_DIRECTORY_2 \
  --output-dir outputs/training --device cuda:0 \
  --pretrained-backbone
```

Omit both Re:InterHand arguments for THAB-only training. This changes the training mixture relative to the paper. The original capture list is not hard-coded: specify the captures you have obtained. Training balances the component datasets, uses the original loss weights and optimizer settings, and saves `best_model.pt` including optimizer/scheduler state. Resume with `--resume outputs/training/best_model.pt`. Use `--epochs 1 --steps-per-epoch 1 --max-val-batches 1 --batch-size 2 --workers 0` for a small setup test, not for a trained model.

## Evaluation, tests and citation

[Reproduction notes](docs/reproduction.md) contain commands for the paper tables and explain legacy handedness, validity and frame-selection conventions.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

Geometry tests run with the base requirements. Model/training tests are skipped unless the detector dependencies are installed. They use generated fixtures and random model weights. See [validation scope](docs/validation.md) for what was actually tested and the remaining validation scope.

Please cite the ACDSA 2027 paper if you use this work. Publication DOI/page numbers will be added when assigned. Author metadata is also provided in [CITATION.cff](CITATION.cff). When using the released model or data, also cite the [Zenodo asset record](https://doi.org/10.5281/zenodo.24315081). This DOI identifies the assets; the paper DOI remains pending.

## License and acknowledgements

The release combines code with **different inherited licenses**. The root [Apache-2.0 license](LICENSE) does not replace the terms of the imported components: `triangulation/` retains its [CC BY-NC 4.0 license](triangulation/LICENSE), and `detector/` retains the original mixed-license acknowledgements and license texts. See [third-party notices](THIRD_PARTY_NOTICES.md) and [detector attribution](detector/UPSTREAM.md).

See the Zenodo record for the model and dataset terms. Third-party datasets and pretrained assets must be obtained under their respective terms.
