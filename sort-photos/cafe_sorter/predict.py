"""Inference: frozen embedder + trained linear classifier -> per-class probabilities."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image

from .config import CLASSIFIER_FILE, METADATA_FILE
from .features import ClipEmbedder, Embedder, Skipped, embed_paths


@dataclass(frozen=True)
class Prediction:
    """Prediction for one image file."""

    path: Path
    label: str
    confidence: float
    probabilities: Dict[str, float]


def load_artifact(model_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load the trained estimator and its JSON metadata from ``model_dir``."""
    classifier_path = model_dir / CLASSIFIER_FILE
    metadata_path = model_dir / METADATA_FILE
    if not classifier_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"No trained model in {model_dir}. Run `photo-sorter train` first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return joblib.load(classifier_path), metadata


class PhotoClassifier:
    """Predicts album probabilities from image pixels only."""

    def __init__(self, embedder: Embedder, estimator: Any, class_names: Sequence[str],
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        self.embedder = embedder
        self.estimator = estimator
        self.class_names = list(class_names)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, model_dir: Path, backbone_dir: Path, device: Optional[str] = None) -> "PhotoClassifier":
        """Build a classifier from a saved artifact and the locally cached backbone."""
        estimator, metadata = load_artifact(model_dir)
        embedder = ClipEmbedder(metadata["backbone"]["name"], backbone_dir, device)
        return cls(embedder, estimator, metadata["class_names"], metadata)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities of shape (n, n_classes), columns ordered like ``class_names``."""
        raw = self.estimator.predict_proba(features)
        probabilities = np.zeros((len(features), len(self.class_names)), dtype=np.float64)
        probabilities[:, self.estimator.classes_] = raw
        return probabilities

    def predict_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return class probabilities for already-decoded images."""
        return self.predict_features(self.embedder.embed(images))

    def to_predictions(self, paths: Sequence[Path], probabilities: np.ndarray) -> List[Prediction]:
        """Pair each path with its most likely class."""
        predictions = []
        for path, row in zip(paths, probabilities):
            best = int(np.argmax(row))
            predictions.append(Prediction(
                path=path,
                label=self.class_names[best],
                confidence=float(row[best]),
                probabilities={name: float(p) for name, p in zip(self.class_names, row)},
            ))
        return predictions

    def predict_paths(self, paths: Sequence[Path], batch_size: int, progress: bool = True,
                      desc: str = "Classifying") -> Tuple[List[Prediction], List[Skipped]]:
        """Predict every readable image in ``paths``; unreadable files are returned as skipped."""
        features, embedded, skipped = embed_paths(paths, self.embedder, batch_size, progress, desc)
        if not embedded:
            return [], skipped
        return self.to_predictions(embedded, self.predict_features(features)), skipped
