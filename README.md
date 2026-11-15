# Café Photo Sorter

A café owner has a folder of mixed photos. This command-line tool copies each one into one of three albums,
**Food & Drinks**, **People** or **Interior**, deciding from the image content alone, never the filename.
It runs entirely on your machine: after setup and a one-time model download, sorting works offline and
the original photos are never modified.

**Headline result:** on 45 real café photos from 35 venue/photographer groups that the model never saw,
the retrained classifier sorts **43 / 45** correctly, up from **25 / 45** for the first version. The first
version put **0 / 15** photos of people into People; the retrained one gets **13 / 15**.

## Quick start

Requires Python 3.11+ and git. Installs CPU-only PyTorch (no GPU needed).

```bash
git clone https://github.com/1brah1m0f/sort_photos_task.git
cd sort_photos_task/sort-photos

python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# install in two steps: CPU builds of torch/torchvision, then the project and its pinned dependencies
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

# one-time download of the CLIP image model (~340 MB) into models/backbone/; the only step that needs internet
python -m clip download-model

# try it on photos shipped with the repo (copied to a folder with no labels in the path)
mkdir -p ~/cafe_photos
cp data/holdout_fresh/*/*.jpg ~/cafe_photos/

# sort them with the included trained classifier (no training needed)
python -m clip sort --input ~/cafe_photos --output ~/cafe_albums
```

To sort your own photos, point `--input` at your folder; add `--recursive` to include sub-folders.
The output folder must not be inside the input folder.

**What you get** in `~/cafe_albums`:

```
cafe_albums/
├── Food & Drinks/      copies of the photos predicted as food or drinks
├── Interior/           copies of the photos predicted as interior
├── People/             copies of the photos predicted as people
└── predictions.csv     one row per copied photo
```

`predictions.csv` columns: `original_path, album, confidence, low_confidence, copied_to, p_food_drinks,
p_people, p_interior`. The first row from the Quick start run above:

```
original_path,album,confidence,low_confidence,copied_to,p_food_drinks,p_people,p_interior
/home/you/cafe_photos/04f929838b0af618a3f0c1bc.jpg,People,0.5839,False,/home/you/cafe_albums/People/04f929838b0af618a3f0c1bc.jpg,0.1707,0.5839,0.2454
```

Files are copied, never moved or renamed. Corrupt and non-image files are skipped with a warning, and name
clashes get `_1`, `_2`, …. Exit status: `0` success, including when some files were skipped; `1` every
attempted copy failed; `2` invalid arguments, unsafe paths or a missing model.

## Results: same 45 fresh café photos, two models

| | First model | Retrained model |
|---|---|---|
| **Overall** | 25 / 45 (55.6%) | **43 / 45 (95.6%)** |
| Food & Drinks: correct (recall) | 10 / 15 | **15 / 15** |
| **People: correct (recall)** | **0 / 15** | **13 / 15** |
| Interior: correct (recall) | 15 / 15 | 15 / 15 |
| Interior precision (how often "Interior" was right) | 0.43 | 0.94 |

The first model sent every café People photo to Interior. That is why its Interior recall looks perfect while
its precision is low: Interior was simply its default answer for café scenes. The retrained model uses the same
frozen CLIP backbone and logistic regression, trained on audited labels plus 171 licensed café photos. None of
the 45 test photos, and none of their venues or photographers, were used for training or model selection.

- Full evaluation, training details, a difficult example and limitations: [`sort-photos/README.md`](sort-photos/README.md#results)
- Reports: [retrained](sort-photos/reports/holdout_fresh/evaluation.md) · [first model](sort-photos/reports/holdout_fresh/baseline/evaluation.md) · [confusion matrix](sort-photos/reports/holdout_fresh/confusion_matrix.png)

Reproduce both columns (from `sort-photos/`, after the Quick start; runs offline):

```bash
python -m clip evaluate --data data/holdout_fresh --report-dir reports/holdout_fresh
python -m clip evaluate --data data/holdout_fresh --report-dir reports/holdout_fresh/baseline --model-dir models/baseline
```

## Tests

```bash
cd sort-photos
python -m pip install -e ".[dev]"
python -m pytest
```

CI runs the same two-step install and the test suite: [`.github/workflows/test.yml`](.github/workflows/test.yml).
