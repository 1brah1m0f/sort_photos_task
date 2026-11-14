"""Training: embed labelled photos with the frozen backbone and fit a logistic-regression head."""
import json
import logging
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .clip import checkpoint_path, expected_sha256
from .config import C_GRID, CLASS_NAMES, CLASSIFIER_FILE, CV_FOLDS, METADATA_FILE, SEED, set_seed
from .features import ClipEmbedder, Embedder, embed_paths
from .fileops import list_files

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """Summary of cross-validation and the final fit."""

    best_c: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_folds: int
    cv_results: List[Dict[str, float]]
    train_counts: Dict[str, int]
    skipped: List[str] = field(default_factory=list)


def collect_labeled_images(data_dir: Path, class_names: Sequence[str] = CLASS_NAMES) -> Tuple[List[Path], np.ndarray]:
    """Return image paths and integer labels from ``data_dir/<class_name>/``.

    Folder names are used only as training/evaluation *labels*; they never reach the model's input.
    """
    paths: List[Path] = []
    labels: List[int] = []
    for index, name in enumerate(class_names):
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        images, _ = list_files(class_dir, recursive=True)
        if not images:
            raise ValueError(f"No images found in {class_dir}")
        paths.extend(images)
        labels.extend([index] * len(images))
    return paths, np.array(labels)


def fit_classifier(features: np.ndarray, labels: np.ndarray, seed: int = SEED, folds: int = CV_FOLDS,
                   c_grid: Sequence[float] = C_GRID) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Choose ``C`` with stratified k-fold CV, then refit on all training data."""
    search = GridSearchCV(
        LogisticRegression(max_iter=5000, random_state=seed),
        param_grid={"C": list(c_grid)},
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        scoring="accuracy",
        refit=True,
    )
    search.fit(features, labels)
    results = search.cv_results_
    table = [
        {"C": float(c), "mean_accuracy": float(m), "std_accuracy": float(s)}
        for c, m, s in zip(results["param_C"], results["mean_test_score"], results["std_test_score"])
    ]
    best = int(search.best_index_)
    summary = {
        "best_c": float(search.best_params_["C"]),
        "cv_accuracy_mean": float(results["mean_test_score"][best]),
        "cv_accuracy_std": float(results["std_test_score"][best]),
        "cv_results": table,
    }
    return search.best_estimator_, summary


def save_model(out_dir: Path, estimator: LogisticRegression, metadata: Dict[str, Any]) -> None:
    """Write ``classifier.joblib`` and ``classifier.json`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, out_dir / CLASSIFIER_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def train_from_embedder(data_dir: Path, embedder: Embedder, batch_size: int, seed: int = SEED,
                        folds: int = CV_FOLDS, progress: bool = True) -> Tuple[LogisticRegression, TrainingReport]:
    """Embed the training images with ``embedder`` and fit the classifier head."""
    set_seed(seed)
    paths, labels = collect_labeled_images(data_dir)
    features, embedded, skipped = embed_paths(paths, embedder, batch_size, progress, desc="Embedding train")
    keep = {p: i for i, p in enumerate(paths)}
    y = labels[[keep[p] for p in embedded]]
    estimator, summary = fit_classifier(features, y, seed=seed, folds=folds)
    counts = {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    report = TrainingReport(cv_folds=folds, train_counts=counts, skipped=[str(p) for p, _ in skipped], **summary)
    return estimator, report


def train(data_dir: Path, out_dir: Path, backbone_name: str, backbone_dir: Path, device: str,
          batch_size: int, seed: int = SEED, folds: int = CV_FOLDS) -> TrainingReport:
    """Full training run: load backbone, embed, cross-validate, fit and save the artifact."""
    embedder = ClipEmbedder(backbone_name, backbone_dir, device)
    estimator, report = train_from_embedder(data_dir, embedder, batch_size, seed, folds)
    metadata = {
        "class_names": list(CLASS_NAMES),
        "backbone": {
            "name": backbone_name,
            "source": "OpenAI CLIP (MIT license), image encoder only, frozen",
            "checkpoint": checkpoint_path(backbone_name, backbone_dir).name,
            "sha256": expected_sha256(backbone_name),
        },
        "classifier": {"type": "LogisticRegression", "C": report.best_c, "max_iter": 5000},
        "training": asdict(report),
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_model(out_dir, estimator, metadata)
    return report
