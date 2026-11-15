"""Training: embed labelled photos with the frozen backbone and fit a logistic-regression head."""
import csv
import json
import logging
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from .clip import checkpoint_path, expected_sha256
from .config import (C_GRID, CLASS_NAMES, CLASS_WEIGHTS, CLASSIFIER_FILE, CV_FOLDS, CV_SCORING, METADATA_FILE, SEED,
                     set_seed)
from .features import ClipEmbedder, Embedder, embed_paths
from .fileops import list_files

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """Summary of cross-validation and the final fit."""

    best_c: float
    best_class_weight: Optional[str]
    cv_score_mean: float
    cv_score_std: float
    cv_scoring: str
    cv_grouped: bool
    n_groups: int
    cv_folds: int
    cv_results: List[Dict[str, Any]]
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


def check_class_counts(counts: Dict[str, int], folds: int, skipped: int = 0) -> None:
    """Make sure every class has enough *readable* images for ``folds``-fold stratified CV.

    Raises
    ------
    ValueError
        If ``folds`` < 2 or any class has fewer than ``folds`` decodable images.
    """
    if folds < 2:
        raise ValueError(f"--folds must be at least 2, got {folds}")
    short = {name: n for name, n in counts.items() if n < folds}
    if short:
        detail = ", ".join(f"{name}: {n}" for name, n in short.items())
        raise ValueError(f"{folds}-fold cross-validation needs at least {folds} readable images per class; "
                         f"got {detail} ({skipped} unreadable training files skipped). "
                         "Add images or lower --folds. No model was written.")


def load_groups(groups_file: Path, paths: Sequence[Path]) -> List[str]:
    """Return the CV group (venue or source collection) of each image.

    ``groups_file`` is a CSV with ``class``, ``filename`` and ``group`` columns; an image is matched by
    its class folder and file name. Groups only decide how cross-validation splits the data.

    Raises
    ------
    ValueError
        If the CSV lacks the columns or any image has no group.
    """
    with open(groups_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"class", "filename", "group"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{groups_file} needs 'class', 'filename' and 'group' columns")
        table = {(row["class"], row["filename"]): row["group"] for row in reader if row["group"]}
    groups, missing = [], []
    for path in paths:
        group = table.get((path.parent.name, path.name))
        if group is None:
            missing.append(f"{path.parent.name}/{path.name}")
        groups.append(group or "")
    if missing:
        raise ValueError(f"{len(missing)} training images have no group in {groups_file} "
                         f"(e.g. {', '.join(missing[:3])}). No model was written.")
    return groups


def fit_classifier(features: np.ndarray, labels: np.ndarray, seed: int = SEED, folds: int = CV_FOLDS,
                   c_grid: Sequence[float] = C_GRID, class_weights: Sequence[Optional[str]] = CLASS_WEIGHTS,
                   groups: Optional[Sequence[str]] = None,
                   scoring: str = CV_SCORING) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Choose ``C`` and ``class_weight`` by cross-validation, then refit on all training data.

    With ``groups`` the folds are stratified *and* grouped (``StratifiedGroupKFold``): all photos of a
    venue or source collection fall in the same fold, so the score measures generalisation to unseen
    venues. Without groups, plain stratified k-fold is used.
    """
    if groups is not None:
        n_groups = len(set(groups))
        if n_groups < folds:
            raise ValueError(f"Grouped {folds}-fold CV needs at least {folds} groups, got {n_groups}.")
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        n_groups = 0
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        LogisticRegression(max_iter=5000, random_state=seed),
        param_grid={"C": list(c_grid), "class_weight": list(class_weights)},
        cv=cv,
        scoring=scoring,
        refit=True,
    )
    search.fit(features, labels, groups=groups)
    results = search.cv_results_
    table = [
        {"C": float(c), "class_weight": w, "mean_score": float(m), "std_score": float(s)}
        for c, w, m, s in zip(results["param_C"], results["param_class_weight"], results["mean_test_score"],
                              results["std_test_score"])
    ]
    best = int(search.best_index_)
    summary = {
        "best_c": float(search.best_params_["C"]),
        "best_class_weight": search.best_params_["class_weight"],
        "cv_score_mean": float(results["mean_test_score"][best]),
        "cv_score_std": float(results["std_test_score"][best]),
        "cv_scoring": scoring,
        "cv_grouped": groups is not None,
        "n_groups": n_groups,
        "cv_results": table,
    }
    return search.best_estimator_, summary


def save_model(out_dir: Path, estimator: LogisticRegression, metadata: Dict[str, Any]) -> None:
    """Write ``classifier.joblib`` and ``classifier.json`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, out_dir / CLASSIFIER_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def train_from_embedder(data_dir: Path, embedder: Embedder, batch_size: int, seed: int = SEED,
                        folds: int = CV_FOLDS, progress: bool = True,
                        groups_file: Optional[Path] = None) -> Tuple[LogisticRegression, TrainingReport]:
    """Embed the training images with ``embedder`` and fit the classifier head.

    Raises
    ------
    ValueError
        If a class has fewer readable images than ``folds`` (checked after decoding, before fitting),
        or ``groups_file`` does not cover every image.
    """
    if folds < 2:
        raise ValueError(f"--folds must be at least 2, got {folds}")
    set_seed(seed)
    paths, labels = collect_labeled_images(data_dir)
    path_groups = dict(zip(paths, load_groups(groups_file, paths))) if groups_file is not None else None
    features, embedded, skipped = embed_paths(paths, embedder, batch_size, progress, desc="Embedding train")
    keep = {p: i for i, p in enumerate(paths)}
    y = labels[[keep[p] for p in embedded]]
    counts = {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    check_class_counts(counts, folds, len(skipped))
    groups = [path_groups[p] for p in embedded] if path_groups is not None else None
    estimator, summary = fit_classifier(features, y, seed=seed, folds=folds, groups=groups)
    report = TrainingReport(cv_folds=folds, train_counts=counts, skipped=[str(p) for p, _ in skipped], **summary)
    return estimator, report


def train(data_dir: Path, out_dir: Path, backbone_name: str, backbone_dir: Path, device: str,
          batch_size: int, seed: int = SEED, folds: int = CV_FOLDS, groups_file: Optional[Path] = None) -> TrainingReport:
    """Full training run: load backbone, embed, cross-validate, fit and save the artifact."""
    embedder = ClipEmbedder(backbone_name, backbone_dir, device)
    estimator, report = train_from_embedder(data_dir, embedder, batch_size, seed, folds, groups_file=groups_file)
    metadata = {
        "class_names": list(CLASS_NAMES),
        "backbone": {
            "name": backbone_name,
            "source": "OpenAI CLIP (MIT license), image encoder only, frozen",
            "checkpoint": checkpoint_path(backbone_name, backbone_dir).name,
            "sha256": expected_sha256(backbone_name),
        },
        "classifier": {"type": "LogisticRegression", "C": report.best_c, "class_weight": report.best_class_weight,
                       "max_iter": 5000},
        "training_data": str(data_dir),
        "groups_file": str(groups_file) if groups_file is not None else None,
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
