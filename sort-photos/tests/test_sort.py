"""Sorter tests using tiny generated images and a colour-based stand-in for the CLIP embedder."""
import csv
import hashlib
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression

from clip.config import ALBUMS, CLASS_NAMES, PREDICTIONS_FILE
from clip.features import open_image
from clip.fileops import unique_destination
from clip.predict import PhotoClassifier
from clip.sorter import sort_folder

# food_drinks -> red, people -> green, interior -> blue
COLOURS = {"food_drinks": (220, 30, 30), "people": (30, 220, 30), "interior": (30, 30, 220)}


class MeanColourEmbedder:
    """Deterministic, offline embedder: the mean RGB of the image."""

    name = "mean-colour"

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        return np.stack([np.asarray(im.resize((8, 8)), dtype=np.float32).mean(axis=(0, 1)) / 255 for im in images])


@pytest.fixture(scope="module")
def classifier() -> PhotoClassifier:
    rng = np.random.default_rng(0)
    centres = np.array([COLOURS[name] for name in CLASS_NAMES], dtype=np.float32) / 255
    X = np.concatenate([c + rng.normal(0, 0.03, (20, 3)) for c in centres])
    y = np.repeat(np.arange(len(CLASS_NAMES)), 20)
    return PhotoClassifier(MeanColourEmbedder(), LogisticRegression(C=100).fit(X, y), CLASS_NAMES)


def make_image(path: Path, label: str, size=(24, 16), **save_kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, COLOURS[label]).save(path, **save_kwargs)
    return path


def tree_hashes(root: Path) -> Dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def photos(tmp_path: Path) -> Path:
    src = tmp_path / "photos"
    make_image(src / "a1b2.jpg", "food_drinks")
    make_image(src / "IMG_0001.JPEG", "people")
    make_image(src / "x.png", "interior")
    make_image(src / "nested" / "q.webp", "food_drinks")
    return src


def test_originals_unchanged(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    before = tree_hashes(photos)
    sort_folder(photos, tmp_path / "out", classifier, recursive=True, progress=False)
    assert tree_hashes(photos) == before


def test_three_albums_created_and_photos_sorted(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    result = sort_folder(photos, out, classifier, recursive=True, progress=False)
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == sorted(ALBUMS.values())
    assert result.counts() == {"Food & Drinks": 2, "People": 1, "Interior": 1}
    assert (out / "People" / "IMG_0001.JPEG").is_file()
    assert (out / "Interior" / "x.png").is_file()


def test_non_recursive_ignores_subfolders(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    result = sort_folder(photos, tmp_path / "out", classifier, recursive=False, progress=False)
    assert len(result.sorted_files) == 3


def test_corrupt_and_non_image_files_skipped(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    (photos / "broken.jpg").write_bytes(b"this is not a jpeg")
    good = make_image(tmp_path / "full.jpg", "people", size=(200, 200))
    (photos / "truncated.jpg").write_bytes(good.read_bytes()[:300])
    (photos / "notes.txt").write_text("hello")
    result = sort_folder(photos, tmp_path / "out", classifier, progress=False)
    skipped = {p.name for p, _ in result.skipped}
    assert {"broken.jpg", "truncated.jpg", "notes.txt"} <= skipped
    assert len(result.sorted_files) == 3


def test_filename_collisions(tmp_path: Path, classifier: PhotoClassifier) -> None:
    src, out = tmp_path / "photos", tmp_path / "out"
    make_image(src / "photo.jpg", "food_drinks")
    make_image(src / "day2" / "photo.jpg", "food_drinks")
    make_image(out / "Food & Drinks" / "photo.jpg", "people")  # left over from an earlier run
    sort_folder(src, out, classifier, recursive=True, progress=False)
    names = sorted(p.name for p in (out / "Food & Drinks").iterdir())
    assert names == ["photo.jpg", "photo_1.jpg", "photo_2.jpg"]
    assert unique_destination(out / "Food & Drinks", "photo.jpg").name == "photo_3.jpg"


def test_output_inside_input_rejected(photos: Path, classifier: PhotoClassifier) -> None:
    with pytest.raises(ValueError):
        sort_folder(photos, photos / "sorted", classifier, progress=False)
    with pytest.raises(ValueError):
        sort_folder(photos, photos, classifier, progress=False)


def test_predictions_csv_and_min_confidence(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    result = sort_folder(photos, out, classifier, recursive=True, min_confidence=1.01, progress=False)
    with open(out / PREDICTIONS_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert set(rows[0]) >= {"original_path", "album", "confidence"} | {f"p_{c}" for c in CLASS_NAMES}
    assert all(row["low_confidence"] == "True" for row in rows)
    assert len(result.low_confidence) == 4


def test_prediction_ignores_filename(tmp_path: Path, classifier: PhotoClassifier) -> None:
    src, out = tmp_path / "photos", tmp_path / "out"
    make_image(src / "food_drinks_pizza.jpg", "interior")  # misleading name, blue pixels
    result = sort_folder(src, out, classifier, progress=False)
    assert result.sorted_files[0].album == "Interior"


def test_exif_orientation_applied(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees when displayed
    path = make_image(tmp_path / "rotated.jpg", "people", size=(40, 20), exif=exif)
    assert open_image(path).size == (20, 40)


def test_backbone_builds_image_encoder_only() -> None:
    import torch

    from clip.model import VisionTransformer, build_model

    tiny = VisionTransformer(input_resolution=32, patch_size=16, width=64, layers=2, heads=1, output_dim=8)
    state = {f"visual.{k}": v for k, v in tiny.state_dict().items()}
    state["text_projection"] = torch.zeros(64, 8)  # text-tower weights must be ignored
    model = build_model(state)
    assert model.encode_image(torch.zeros(2, 3, 32, 32)).shape == (2, 8)
