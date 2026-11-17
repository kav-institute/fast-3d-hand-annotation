# License provenance

The target repository's existing Apache-2.0 `LICENSE` is preserved. Imported material has not been relicensed.

- `triangulation/src/`: CC BY-NC 4.0 license is preserved at `triangulation/LICENSE`.
- `detector/src/`: Its original acknowledgement is preserved at `detector/UPSTREAM.md`; it identifies material rewritten, adapted or inspired by InterHand2.6M, InterWild, Sapiens and MediaPipe, with CC BY-NC 4.0 and Apache-2.0 terms. The supplied license texts are retained in `detector/LICENSES/`. The source does not provide a per-function provenance map, so the combined detector should not be described as Apache-only.
- The inherited detector configuration remains under `detector/src/data/`; this directory contains Python configuration only, not model weights.

The synthetic fixture generator and tests contain no research recordings. Model checkpoints, research datasets, rendered assets and third-party pretrained weights are not bundled. See the [Zenodo asset record](https://doi.org/10.5281/zenodo.24315081) for asset terms, separately from the code licenses.
