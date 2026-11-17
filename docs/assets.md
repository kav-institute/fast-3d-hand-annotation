# Models, assets and data

The model and scene-data bundle is available on [Zenodo: 10.5281/zenodo.24315081](https://doi.org/10.5281/zenodo.24315081). Download both ZIP archives and the accompanying manifest, README, unpacking helper and `SHA256SUMS`. The third-party hand detector and Sapiens model are already available upstream: see the [download instructions](../README.md#download-pretrained-models). The paths below are the local layout used by the command examples.

| Local path | Needed for | Expected content |
| --- | --- | --- |
| `models/hand_keypoints.pt` | Learned 2D inference | `MultiHandKeypoint(21)` state dict, either directly or in `model_state_dict`; strict key/shape matching |
| `models/hand_detector.pth` | Automatic hand bounding boxes | Weights matching `detector/src/data/cascade_rcnn_x101_64x4d_fpn_1class.py` |
| `models/sapiens_wholebody.pt2` | Optional Sapiens comparison | Sapiens-2B COCO WholeBody TorchScript model with 133 heatmaps |
| `data/<scene>/` | Lifting/evaluation | Calibration, 2D candidates and ground truth; images needed only for inference/visualization |
| `data/thab_training/` | Training | Training and validation annotations, images and canonical skeleton |
| `data/reinterhand/` | Optional mixed-dataset training | Original separately acquired Re:InterHand captures and fits |

The bundle uses `scenes/<scene>/`; link or copy these directories to `data/<scene>/` for the examples above. See the [asset setup and InterHand camera guide](interhand-setup.md) for checksum verification, exact selections and optional image restoration. Only our learned checkpoint is bundled; obtain the two third-party checkpoints separately.

Existing candidate tensors are sufficient for CPU triangulation; model weights are not needed for that workflow. Ground truth is only needed for evaluation.
