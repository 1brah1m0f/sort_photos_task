# Label audit

Labelling rule from the assignment: each photo goes to the album of its **main subject**
(Food & Drinks, People or Interior).

## Changes

| File | Split | Old label | New label | Date |
|---|---|---|---|---|
| `337d124eeda7482b9aba84aae4ef9528.jpg` | test | interior | food_drinks | 2026-11-15 |

**Reason.** The source is Places365 val, scene category *coffee_shop*. Places365 labels the **scene**
a photo was taken in, not its subject. The photo is a close-up of a bakery display case. Sugar-dusted
bomboloni and other pastries fill almost the whole frame, with price cards in front. The only
"interior" visible is a strip of glass counter. By the main-subject rule this is Food & Drinks.
The file was moved from `test/interior/` to `test/food_drinks/` and its row in `manifest.csv` was updated.

**Effect on results.** The classifier is unchanged: it never saw test labels, and `C` is chosen by
cross-validation on `data/train` only. The model predicted Food & Drinks (0.92) for this photo
both before and after. The original benchmark split therefore scores 44/45 with the old label and
45/45 with the corrected one.

## Known, unchanged label noise

- **Training data is scene-labelled too.** All 50 `train/interior` images are Places365 *coffee_shop*
  scenes, so some contain mostly a cup, a counter or people. For example
  `train/interior/27b610fb51eb4d6da6950d1e9fb57b73.jpg` is a close-up of a coffee cup. Training labels
  were **not** changed: the classifier and its model selection stay fixed, so the held-out café
  evaluation in `holdout_cafe/` measures the model as shipped.
- Only the one test image above was audited individually. The remaining train/test labels follow their
  source datasets (Food-101 and Wikimedia drink categories → food_drinks, COCO → people,
  Places365 coffee_shop → interior).

## Held-out café set (`holdout_cafe/`)

Labels were assigned by the main-subject rule from the images alone, **before** any prediction was run
(timestamp in `holdout_cafe/manifest.csv`, column `labeled_at_utc`). Photos where a second subject is
prominent are marked `mixed_subject=yes`, and `label_reason` explains the choice.

The labels (and this audit) were made by one annotator, the AI coding assistant that built the set, from
visual inspection, with no second reviewer. A café owner might judge some mixed scenes differently. To
re-evaluate with your own labels, move files between the class folders and re-run
`python -m clip evaluate --data data/holdout_cafe --report-dir reports/holdout_cafe`.
