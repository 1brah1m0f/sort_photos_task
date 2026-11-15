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

# sort a folder (the trained classifier in models/ is included in the repo)
python -m clip sort --input /path/to/photos --output /path/to/albums --recursive
```

`/path/to/albums` now contains `Food & Drinks/`, `People/`, `Interior/` and `predictions.csv`.
The output folder must not be inside the input folder.

## Retrain

```bash
cd sort-photos
python -m clip train --data data/train --out models/
```

## Evaluate

```bash
cd sort-photos
python -m clip evaluate --data data/test                                             # benchmark split
python -m clip evaluate --data data/holdout_cafe --report-dir reports/holdout_cafe   # real café holdout
```

Current results: 45 / 45 on the benchmark split after one label correction (44 / 45 with its original
labels), but only **26 / 45 on real café photos**. Café photos of people are mostly sorted into Interior.
See [Results](sort-photos/README.md#results).

## Tests

```bash
cd sort-photos
python -m pytest
```

CI runs the same two-step install and the tests: [`.github/workflows/test.yml`](.github/workflows/test.yml).
