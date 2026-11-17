# Validation of the combined release

The release was exercised on Linux / Python 3.12 with CPU PyTorch. Tests use temporary files; neither source repository nor the research datasets is modified.

Tested versions: Python 3.12.3, NumPy 1.26.4, SciPy 1.11.4, OpenCV 4.6.0, Pillow 10.2.0, Matplotlib 3.6.3, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, timm 1.0.31 and h5py 3.16.0. These record the test environment, not the historical paper training environment.

## Automated checks

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

**15 tests passed**, covering:

- Noise-free calibrated multi-view reconstruction through all five lifting variants.
- ANT with noisy candidates, fully missing joints, temporal/anatomical refinement and graph optimization.
- Rejection of inconsistent calibration, malformed/non-finite detections, unequal frame counts and mismatched image names.
- Camera IDs containing underscores and evaluation with entirely missing predictions.
- A real HRNet-W48/SimCC forward pass, crop mapping and export/import of the candidate tensor format.
- Training loss/backpropagation with generated inputs.
- A complete one-step training/validation/checkpoint-saving run using generated THAB-format annotations and images.

The learned model tests use **random weights**, not the paper's trained checkpoint. They validate code execution and interfaces, not trained accuracy.

All **17 executable CLI entry points** passed `--help` from outside the repository directory. All Python sources passed compilation.

## Comparisons with the original code

Using local data kept outside the release, ANT was run with the original and merged code on two frames, up to six cameras and the first 100 candidates from each of:

- `scene_2_light`
- `scene_3_rom02_interaction_2_hand`
- `2025_11_13_144123_pp_duplo_interation_p001_16fps_s001_7a5488a5`

All output arrays matched numerically, including validity and residuals (84 valid joints per sampled scene). This is a regression comparison on small subsets, not a full remeasurement of accuracy or runtime.

The two InterHand 3D tables, workplace 3D table and workplace refinement table were also generated from existing archived outputs with both versions. All four LaTeX tables were identical. This checks the merged evaluation code against the original; it does not independently validate the archived predictions or the published claims.

## Release cleanup verification

After removing dead code and unused settings/dependencies, all 15 tests passed again. All 18 CLI entry points (including the InterHand setup helper) passed `--help`, and Ruff's Pyflakes checks found no unused imports or undefined names.

The trained checkpoint still loaded with strict state-dict matching. Its CPU predictions and one real sample from each Re:InterHand rig (Mugsy and Ego) matched the pre-cleanup version exactly across all 36 compared arrays. The recorded InterHand scene's evaluator output was also identical before and after cleanup. These are targeted regression checks, not full-dataset accuracy validation.

## Remaining validation scope

- GPU/CUDA execution, the optional MMDetection native operators, live camera preview, and real Sapiens/MediaPipe inference were not exercised in this environment.
- Full mixed-dataset training and complete scene inference with the paper checkpoint have not been rerun. The trained checkpoint was separately loaded with strict state-dict matching and passed a CPU forward-pass smoke test; this does not measure trained accuracy. Re:InterHand loading was made configurable but was not run against the full original training captures.
- No complete dataset rerun or performance benchmark was performed. Do not use the short CPU smoke tests as evidence for the paper's throughput numbers.
- End-to-end reconstruction from recorded images with the trained model remains to be validated against the published bundle on [Zenodo: 10.5281/zenodo.24315081](https://doi.org/10.5281/zenodo.24315081). Publication of the assets does not extend the test coverage reported above.

The source snapshot hashes and component commit IDs are recorded in [source-manifest.json](source-manifest.json). The upstream commits may contain additional history; only the selected Python sources and license notices were imported.
