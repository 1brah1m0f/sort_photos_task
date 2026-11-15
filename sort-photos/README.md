# Café Photo Sorter

A small command-line tool that sorts a café owner's folder of mixed photos into three albums,
**Food & Drinks**, **People** and **Interior**, based only on what is in each picture.
Originals are never touched: every photo is *copied* into its predicted album.

Everything runs locally in Python. After a one-time model download the tool works fully offline.

## Approach

```
photo ──► EXIF-upright RGB ──► CLIP ViT-B/32 image encoder (frozen) ──► 512-d embedding ──► logistic regression ──► album
```

1. **Feature extractor**: the image tower of OpenAI's open-source CLIP ViT-B/32 (MIT license),
   used frozen. Each photo becomes an L2-normalised 512-dimensional embedding.
2. **Classifier**: a scikit-learn `LogisticRegression` trained on those embeddings. Its
   regularisation strength `C` is chosen with 5-fold stratified cross-validation on the training set.

**Why this works with only 50 photos per class.** CLIP was pretrained on ~400M image–text pairs,
so its embeddings already separate concepts like "a plate of food", "a person" and "a room".
Fine-tuning a deep network on 150 images would overfit badly. A linear model on frozen features has
only 3 × 513 parameters, trains in under a second, and only needs to find the directions in
embedding space that the backbone already encodes. This is the standard "linear probe" setup.

**Only pixels are used.** Predictions depend on decoded pixel data alone. Filenames, folder names
and EXIF text fields are never read for classification. The EXIF *orientation* tag is applied so
that rotated phone photos are seen upright. During training and evaluation, folder names are used
only as labels. A test (`test_prediction_ignores_filename`) checks this.

## Setup

Requires Python ≥ 3.10.

```bash
cd sort-photos
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU-only torch: add --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"                  # optional: installs the `photo-sorter` command and pytest
```

A GPU is used automatically when available (`--device cpu` forces the CPU).

## Usage

All commands are run from the `sort-photos/` folder. `python -m clip <command> --help` lists all options.

```bash
# 1. One-time download of the backbone weights (~340 MB) into models/backbone/ (SHA256-verified).
#    This is the only command that uses the network.
python -m clip download-model

# 2. Train the classifier on data/train/<class>/ -> models/classifier.joblib + models/classifier.json
python -m clip train --data data/train --out models/

# 3. Evaluate on held-out photos -> console metrics + reports/evaluation.md + reports/confusion_matrix.png
python -m clip evaluate --data data/test

# 4. Sort your own photos
python -m clip sort --input ~/Pictures/cafe --output ~/Pictures/cafe_albums [--recursive] [--min-confidence 0.7]
```

What `sort` does:

- creates `Food & Drinks/`, `People/` and `Interior/` inside `--output` and **copies**
  (`shutil.copy2`) each photo into its predicted album. Originals are never moved, renamed or modified.
- refuses an output folder that is the input folder or inside it.
- accepts `.jpg .jpeg .png .webp .bmp .tif .tiff` in any letter case, and applies EXIF orientation.
- skips corrupt or non-image files with a warning instead of crashing.
- resolves name collisions by appending `_1`, `_2`, … (for example with `--recursive`, or when re-running into the same output).
- classifies in batches (`--batch-size`, default 32) with a progress bar.
- writes `predictions.csv` (`original_path, album, confidence, low_confidence, copied_to, p_food_drinks, p_people, p_interior`).
- with `--min-confidence`, photos whose top probability is below the threshold are still copied to
  their best album, but are flagged in the CSV and listed in the console summary.
- prints a summary: photos per album, low-confidence photos, skipped files.

### Offline use

`download-model` stores the checkpoint at `models/backbone/ViT-B-32.pt`. `train`, `evaluate` and
`sort` load it **only** from that path. They never download anything and fail with a clear message
if the file is missing. They also set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. This was
verified by running `sort` with the network removed at the OS level:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 unshare -rn python -m clip sort --input demo_in --output demo_out
```

### Tests

```bash
python -m pytest
```

The tests use tiny generated images and a colour-based stand-in for CLIP, so they run in seconds
without the backbone. They cover: originals unchanged after sorting (SHA256 of every file), the
three albums being created, corrupt/truncated/non-image files being skipped, filename collisions,
output-inside-input rejection, the predictions CSV and the low-confidence flag, filename independence,
EXIF orientation, and building the image encoder from a checkpoint.

## Project structure

```
sort-photos/
├── clip/                    # the package (python -m clip ...)
│   ├── __main__.py          # entry point
│   ├── cli.py               # argparse CLI: download-model / train / evaluate / sort
│   ├── config.py            # classes, album names, paths, seeds, device
│   ├── clip.py              # backbone download (only network use) + local-only loading + preprocessing
│   ├── model.py             # CLIP ViT image encoder (text tower removed)
│   ├── features.py          # image decoding (EXIF orientation, corrupt-file handling) + batched embedding
│   ├── train.py             # stratified CV over C, final fit, saving the model artifact
│   ├── predict.py           # PhotoClassifier: embeddings -> class probabilities
│   ├── evaluate.py          # metrics, confusion matrix, Markdown/JSON report
│   ├── sorter.py            # sort a folder: classify, copy into albums, write predictions.csv
│   └── fileops.py           # file discovery, folder validation, collision-safe copying
├── data/                    # train/ and test/ with food_drinks/, people/, interior/ (50 + 15 each) + manifest.csv (sources, licenses)
├── models/                  # classifier.joblib, classifier.json (classes, backbone name + SHA256, C, CV scores, versions); backbone/ is git-ignored
├── reports/                 # evaluation.md/json, confusion_matrix.png, difficult_example.jpg
├── tests/test_sort.py
├── requirements.txt         # pinned
└── setup.py
```

## Results

Backbone CLIP ViT-B/32 (frozen) + logistic regression, seed 42, CPU.

**Cross-validation on the training set** (150 images, 5-fold stratified):

| C | 0.01 | 0.1 | 1 | 10 | **100** | 1000 |
|---|---|---|---|---|---|---|
| mean accuracy | 0.913 | 0.913 | 0.927 | 0.940 | **0.947** | 0.947 |

`C = 100` was selected (ties are broken in favour of the first, i.e. the smaller, value).

**Held-out test set** (45 images not used for training):

### 44 / 45 test images correctly classified (97.8%)

| Album | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Food & Drinks | 0.938 | 1.000 | 0.968 | 15 |
| People | 1.000 | 1.000 | 1.000 | 15 |
| Interior | 1.000 | 0.933 | 0.966 | 15 |

![Confusion matrix](reports/confusion_matrix.png)

Full report: [reports/evaluation.md](reports/evaluation.md).

**Demo simulation.** The 45 test images were copied into a flat folder with random filenames
(for example `1d878f9f9cdf.JPG`), along with a corrupt `.jpg` and a `.txt` file. `sort` was then run
with networking disabled. Result: Food & Drinks 16, People 15, Interior 14; the 2 bad files
were skipped; all originals were byte-identical afterwards. Matching the copies back to their
source files by hash gives **44 / 45 correct**, the same as `evaluate`, which confirms that
predictions do not depend on filenames or folders.

## A difficult example

![Difficult example](reports/difficult_example.jpg)

`data/test/interior/337d124eeda7482b9aba84aae4ef9528.jpg`, labelled **Interior** (it comes from the
Places365 *coffee_shop* scene category), predicted **Food & Drinks**:

| Food & Drinks | People | Interior |
|---|---|---|
| **0.92** | 0.00 | 0.08 |

This is the only test error. The photo is a close-up of a bakery display case: sugar-dusted
bomboloni and pastries fill almost the whole frame, with price cards in front. The "interior" is
only a sliver of glass counter. The label comes from the scene the photo was taken in, but by
*main subject*, the criterion the albums use, it is mostly food. The model's answer is arguably
the more useful one. The case shows the real ambiguity of the task: a café photo often has
two plausible subjects, and the label depends on what the photographer meant to show.

The next hardest photo is a correct prediction: `data/test/people/4eda754771ad4d60ba25939c7b1efa9f.jpg`
got People 0.69 / Interior 0.30. It is a vintage black-and-white snapshot of two people on an armchair
in a living room, with striped curtains, cushions and furniture all around. The people are clearly the subject,
but so much of the furnished room is visible that part of the probability goes to Interior. With `--min-confidence 0.7` it was the
one photo flagged in the demo run.

## Limitations and possible improvements

- **Small test set.** 45 images means each error moves accuracy by 2.2 points. The 97.8% figure has
  wide uncertainty, and CV on the training set (94.7% ± 3.4%) is a more cautious estimate.
- **Single-label albums.** Every photo goes to exactly one album. A barista holding a latte is both
  "People" and "Food & Drinks". Possible fixes: multi-label output (copy to every album above a
  threshold) or an "Unsure" album for low-confidence photos.
- **Domain shift.** The training photos come from public datasets (Food-101, COCO, Places365,
  Wikimedia Commons; see `data/manifest.csv`). The owner's own photos (lighting, phone camera, their
  café) may look different. Adding 10–20 of their own labelled photos per album and re-running
  `train` is the cheapest way to improve.
- **Fixed classes.** A photo of the menu board, the storefront or a dog is still forced into one of
  the three albums. A fourth "Other" class or an embedding-distance reject option would help.
- **Bigger backbones.** `--backbone ViT-B/16` or `ViT-L/14` (with `download-model --backbone ...`)
  usually give better embeddings, at the cost of speed and download size. DINOv2 or SigLIP would also
  fit behind the same `Embedder` interface.
- **Calibration.** Logistic-regression probabilities with a large `C` tend to be over-confident (the error
  above had 0.92). Choosing `--min-confidence` from a validation set, or temperature scaling, would make the
  flag more meaningful.

## License and credits

The image-encoder code in `clip/model.py` and `clip/clip.py` is adapted from
[OpenAI CLIP](https://github.com/openai/CLIP) (MIT, see `LICENSE`). The dataset sources and licenses
of every photo are listed in `data/manifest.csv`.
