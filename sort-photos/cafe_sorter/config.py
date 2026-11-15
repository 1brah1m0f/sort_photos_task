"""Project-wide constants: classes, album names, default paths and seeds."""
import random
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np
import torch

# Class identifiers, in the fixed order used for probability columns everywhere.
CLASS_NAMES: Tuple[str, ...] = ("food_drinks", "people", "interior")

# Human-readable album folder created for each class.
ALBUMS: Dict[str, str] = {
    "food_drinks": "Food & Drinks",
    "people": "People",
    "interior": "Interior",
}

# Optional album for photos below --min-confidence when --unsure is given.
UNSURE_ALBUM = "Unsure"

# Frozen feature extractor (one of cafe_sorter.clip.available_models()).
BACKBONE_NAME = "ViT-B/32"

# Project folder (sort-photos/). Defaults are absolute paths under it, so the CLI works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "train"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_BACKBONE_DIR = DEFAULT_MODEL_DIR / "backbone"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

CLASSIFIER_FILE = "classifier.joblib"
METADATA_FILE = "classifier.json"
PREDICTIONS_FILE = "predictions.csv"

# Extensions accepted by the sorter (compared case-insensitively).
IMAGE_EXTENSIONS: FrozenSet[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"})

BATCH_SIZE = 32
SEED = 42
CV_FOLDS = 5
C_GRID: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
CLASS_WEIGHTS: Tuple[Optional[str], ...] = (None, "balanced")
# Model selection metric: mean per-class recall, so a collapsed class (e.g. People) is not hidden by the others.
CV_SCORING = "balanced_accuracy"
# Manifest next to the training folder; its ``group`` column (venue or source collection) drives grouped CV.
MANIFEST_FILE = "manifest.csv"


def set_seed(seed: int = SEED) -> None:
    """Seed every random number generator used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_groups_file(train_dir: Path) -> Optional[Path]:
    """``<train_dir>/../manifest.csv`` if it exists and has a ``group`` column, else None."""
    candidate = train_dir.parent / MANIFEST_FILE
    if not candidate.is_file():
        return None
    with open(candidate, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return candidate if "group" in header else None


def default_device() -> str:
    """Return ``"cuda"`` when a GPU is available, otherwise ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"
