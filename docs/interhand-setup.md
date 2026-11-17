# Asset setup and InterHand camera selection

## Install the asset bundle

Download the asset bundle from [Zenodo: 10.5281/zenodo.24315081](https://doi.org/10.5281/zenodo.24315081). Download `models.zip`, `scenes.zip`, `interhand-scenes.json`, `README.md`, `unpack_and_verify.py`, and `SHA256SUMS` into one directory. The ZIPs extract to `models/hand_keypoints.pt` and `scenes/`. It contains our learned model, candidate tensors, calibration, ground truth and saved reconstructions. The workplace scene includes its images; **InterHand images are deliberately excluded**. Obtain the hand bounding-box detector and optional Sapiens model using the [README download instructions](../README.md#download-pretrained-models).

First unpack and verify the bundle, before restoring images or changing any files (Python 3.8+, no extra packages):

```bash
cd /absolute/path/to/asset-bundle
python3 unpack_and_verify.py
```

The helper retains both ZIPs and verifies every file listed in `SHA256SUMS`. Use `python3 unpack_and_verify.py --check-only` for later checks. Alternatively, extract both ZIPs into this directory and run `sha256sum -c SHA256SUMS`. The detailed setup guide is maintained here in GitHub; the Zenodo README links to it.

Then, from the code repository root, link the bundle into the paths used by the README and evaluation examples:

```bash
ASSET_ROOT=/absolute/path/to/asset-bundle
mkdir -p data models
ln -s "$ASSET_ROOT"/scenes/* data/
ln -s "$ASSET_ROOT/models/hand_keypoints.pt" models/hand_keypoints.pt
```

Use an absolute `ASSET_ROOT`. These commands expect the destination scene/model paths not to exist. You can also copy the directories or pass the bundle's scene paths directly to `--scene` / `--detection-source-path` / `--scene-root`.

**Images are unnecessary for triangulation and evaluation of the supplied candidates/results.** Restoring them is only needed for fresh 2D inference or image-based visualization.

## Exact InterHand subsets

These are two sequences from the **InterHand2.6M 5 fps image release**, separate from the Re:InterHand training-data setup. Download the original images through the [official dataset page](https://mks0601.github.io/InterHand2.6M/) and follow its download instructions and dataset terms. The [official releases](https://github.com/facebookresearch/InterHand2.6M/releases) also list the 5 fps image archives. Do not use a differently resized or cropped image release with the supplied calibration.

| Paper | Released scene | Original path relative to `images/` | Frames | Cameras |
| --- | --- | --- | --- | --- |
| Scene 1 | `scene_2_light` | `train/Capture0/0390_dh_touchROM` | 168 | 13 |
| Scene 2 | `scene_3_rom02_interaction_2_hand` | `test/Capture0/ROM02_Interaction_2_Hand` | 178 | 8 |

Scene 1 originates from the upstream **train** split; Scene 2 originates from the upstream **test** split. They are the paper's selected evaluation sequences, not both official test-split sequences.

**Scene 1 camera IDs:**

```text
400002 400009 400012 400017 400018 400023 400026
400028 400030 400041 400042 400060 400063
```

**Scene 2 camera IDs:**

```text
400265 400270 400274 400275 400281 400287 400288 400294
```

The [machine-readable selection](interhand-scenes.json) lists every camera and frame ID. It is also included at the asset bundle root. Candidate row `f` corresponds to entry `f` in that scene's `frame_ids` list. This agrees with numerically sorted ground-truth keys and lexicographically sorted image filenames for both selections.

- Scene 1 spans frame IDs **80412–83364**, but its 168-frame selection has gaps of both 9 and 18. Use the explicit list; do not reconstruct it using a fixed stride.
- Scene 2 contains frame IDs **21293–22355**, every 6 frames (178 frames).
- Preserve original `image<frame_id>.jpg` filenames and image dimensions **334 pixels wide × 512 high**. No resizing, cropping or recompression is needed.
- Map upstream `cam<ID>/image<frame_id>.jpg` to released `camera_<ID>/image<frame_id>.jpg`.
- The calibration JSON contains more cameras than the selected subset (73 entries for Scene 1, 13 for Scene 2). Only the IDs listed above participate in the released candidate tensors. Do not use every calibration entry as an image-camera selection.
- `scene_3_rom02_interaction_2_hand_org` is a 13-camera variant of the same Scene 2 sequence, not a third scene. It is not included in this bundle; its results differ from those of the selected eight cameras.

The selections were checked against all 3,608 original local JPEGs: filenames, frame order and image bytes match the source sequences above. Candidate frame counts also match the manifest.

## Restore images from your own download

The repository includes a standard-library-only setup script. `--images-root` must point to the original `images/` directory containing `train/` and `test/`, rather than the archive root or a single capture.

First validate the complete selection without creating files:

```bash
python scripts/prepare_interhand_images.py \
  --images-root /absolute/path/to/InterHand2.6M/images \
  --scenes-root data --dry-run
```

Then create links to exactly the selected images:

```bash
python scripts/prepare_interhand_images.py \
  --images-root /absolute/path/to/InterHand2.6M/images \
  --scenes-root data
```

The script validates every source image, ground-truth frame list, selected calibration ID and destination before writing. It defaults to absolute symlinks, so keep the original download in place. Add `--copy` for independent copies, or `--scene scene_2_light` to restore one scene only. Matching existing images are left in place; conflicting files and unexpected cameras/frames are rejected. It does not download or redistribute images, modify calibration, or regenerate detections.

If you linked `data/` to the asset bundle as above, restoring images adds them to your local bundle directories. These personal image copies/links are not part of the published checksum list or the image-free InterHand release. Keep the original upload bundle separate if preparing a new distribution.

After restoration, a scene looks like:

```text
data/scene_2_light/
  camera_parameter_scene_2.json
  gt_keypoints_3d.json
  hand_skeleton.txt
  detections_camera_400002_2d.npy
  ...
  camera_400002/image80412.jpg
  camera_400002/image80430.jpg
  ...
```

To inspect the supplied results without images:

```bash
python triangulation/src/paper_results_3d_interhand.py \
  --scene-root data/scene_2_light --output /tmp/interhand_scene1.tex
python triangulation/src/paper_results_3d_interhand.py \
  --scene-root data/scene_3_rom02_interaction_2_hand --output /tmp/interhand_scene2.tex
```

For fresh triangulation, see the [README](../README.md#triangulation-and-refinement). For fresh 2D inference, download the required models and follow its [candidate-generation instructions](../README.md#generate-2d-candidates). Existing candidate files are protected unless `--overwrite` is explicitly supplied.

## Archived results and checksums

The archived Scene 2 table values match the manuscript. Some inspected Scene 1 baseline/ANT entries differ by up to 0.002; the historical evaluation mask/settings need confirmation before claiming exact reproduction of every rounded manuscript value. See [reproduction notes](reproduction.md) for evaluation conventions.
