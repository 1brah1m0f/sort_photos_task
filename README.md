# sort_photos_task

A local Python CLI that copies a folder of mixed café photos into three albums: **Food & Drinks**,
**People** and **Interior**. It decides from image content only (frozen CLIP ViT-B/32 embeddings +
logistic regression), never touches the originals, and runs offline after a one-time model download.

The project lives in [`sort-photos/`](sort-photos/). Full documentation, results and limitations are in
[`sort-photos/README.md`](sort-photos/README.md).

## Quickest path: setup to sorted albums

Requires Python 3.11+. CPU install, in two steps:

```bash
cd sort-photos
python3.11 -m venv .venv && source .venv/bin/activate

# step 1: CPU builds of the pinned torch/torchvision from the PyTorch index
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
# step 2: the project and the remaining pinned dependencies from PyPI
python -m pip install -e ".[dev]"

# one-time download of the CLIP weights (~340 MB) into models/backbone/ (the only step that needs internet)
python -m clip download-model

# sort a folder (the trained classifier in models/ is included in the repo; no training needed)
python -m clip sort --input /path/to/photos --output /path/to/albums --recursive
```

`/path/to/albums` now contains `Food & Drinks/`, `People/`, `Interior/` and `predictions.csv`.
The output folder must not be inside the input folder.

## Retrain

```bash
cd sort-photos
python -m clip train --data data/train --out models/   # venue-grouped CV over C × class_weight (groups: data/manifest.csv)
```

## Evaluate (reproduces the numbers below)

```bash
cd sort-photos
python -m clip evaluate --data data/holdout_fresh --report-dir reports/holdout_fresh                                   # retrained
python -m clip evaluate --data data/holdout_fresh --report-dir reports/holdout_fresh/baseline --model-dir models/baseline  # shipped baseline
python -m clip evaluate --data data/holdout_cafe  --report-dir reports/holdout_cafe                                    # diagnostic set
python -m clip evaluate --data data/test                                                                               # benchmark split
```

## Results

The default model in `sort-photos/models/` was retrained on audited labels plus 171 licensed café photos
from 88 venue/photographer groups. The first shipped classifier is kept in `models/baseline/` for comparison.

| Set | Baseline | Retrained |
|---|---|---|
| **Fresh café holdout** (45 photos, 35 venue groups absent from training) | 25 / 45 | **43 / 45** |
| People recall on the fresh holdout | 0 / 15 | **13 / 15** |
| Food & Drinks / Interior recall on the fresh holdout | 10 / 15, 15 / 15 | 15 / 15, 15 / 15 |
| `holdout_cafe` (diagnostic, inspected earlier) | 26 / 45 | 43 / 45 |
| Benchmark split, audited labels (original scene labels) | 41 / 45 (45 / 45) | 44 / 45 (40 / 45) |

Details, the label audit and a difficult example: [`sort-photos/README.md#results`](sort-photos/README.md#results).

## Tests

```bash
cd sort-photos
python -m pytest
```

CI runs the same two-step install and the tests: [`.github/workflows/test.yml`](.github/workflows/test.yml).
