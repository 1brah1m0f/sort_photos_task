"""Shared test helpers: tiny generated images and a colour-based stand-in for the CLIP embedder."""
import hashlib
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression

from cafe_sorter.config import CLASS_NAMES
from cafe_sorter.predict import PhotoClassifier

# food_drinks -> red, people -> green, interior -> blue
COLOURS = {"food_drinks": (220, 30, 30), "people": (30, 220, 30), "interior": (30, 30, 220)}


class MeanColourEmbedder:
    """Deterministic, offline embedder: the mean RGB of the image."""

    name = "mean-colour"

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        return np.stack([np.asarray(im.resize((8, 8)), dtype=np.float32).mean(axis=(0, 1)) / 255 for im in images])


@pytest.fixture(scope="session")
def classifier() -> PhotoClassifier:
    """A PhotoClassifier trained on noisy mean colours (red=food, green=people, blue=interior)."""
    rng = np.random.default_rng(0)
    centres = np.array([COLOURS[name] for name in CLASS_NAMES], dtype=np.float32) / 255
    X = np.concatenate([c + rng.normal(0, 0.03, (20, 3)) for c in centres])
    y = np.repeat(np.arange(len(CLASS_NAMES)), 20)
    return PhotoClassifier(MeanColourEmbedder(), LogisticRegression(C=100).fit(X, y), CLASS_NAMES)


def make_image(path: Path, label: str, size=(24, 16), **save_kwargs) -> Path:
    """Write a solid-colour image whose colour encodes ``label``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, COLOURS[label]).save(path, **save_kwargs)
    return path


def tree_hashes(root: Path) -> Dict[str, str]:
    """SHA256 of every regular file under ``root``, keyed by relative path."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def photos(tmp_path: Path) -> Path:
    """Four images with unrelated filenames, one of them in a sub-folder."""
    src = tmp_path / "photos"
    make_image(src / "a1b2.jpg", "food_drinks")
    make_image(src / "IMG_0001.JPEG", "people")
    make_image(src / "x.png", "interior")
    make_image(src / "nested" / "q.webp", "food_drinks")
    return src


@pytest.fixture
def cli_with_classifier(classifier: PhotoClassifier, monkeypatch: pytest.MonkeyPatch):
    """Run the real CLI (``cafe_sorter.cli.main``) with the colour classifier instead of loading CLIP weights."""
    from cafe_sorter.cli import main

    monkeypatch.setattr(PhotoClassifier, "load", lambda *args, **kwargs: classifier)
    return main
