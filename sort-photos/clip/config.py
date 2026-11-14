"""Project-wide constants: classes, album names, default paths and seeds."""
import random
from pathlib import Path
from typing import Dict, FrozenSet, Tuple

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

# Frozen feature extractor (one of clip.available_models()).
BACKBONE_NAME = "ViT-B/32"

# Default locations, relative to the directory the CLI is run from (the repo root).
DEFAULT_TRAIN_DIR = Path("data/train")
DEFAULT_TEST_DIR = Path("data/test")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_BACKBONE_DIR = DEFAULT_MODEL_DIR / "backbone"
DEFAULT_REPORT_DIR = Path("reports")

CLASSIFIER_FILE = "classifier.joblib"
METADATA_FILE = "classifier.json"
PREDICTIONS_FILE = "predictions.csv"

# Extensions accepted by the sorter (compared case-insensitively).
IMAGE_EXTENSIONS: FrozenSet[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

BATCH_SIZE = 32
SEED = 42
CV_FOLDS = 5
C_GRID: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def set_seed(seed: int = SEED) -> None:
    """Seed every random number generator used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device() -> str:
    """Return ``"cuda"`` when a GPU is available, otherwise ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"
