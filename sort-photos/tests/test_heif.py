"""HEIC/HEIF photos (the iPhone default) are read and sorted like any other image."""
from pathlib import Path

import pytest

import cafe_sorter.features
from cafe_sorter.config import IMAGE_EXTENSIONS
from cafe_sorter.predict import PhotoClassifier
from cafe_sorter.sorter import sort_folder
from conftest import make_image, tree_hashes

pytest.importorskip("pillow_heif")


@pytest.fixture
def heic_photos(tmp_path: Path) -> Path:
    src = tmp_path / "phone"
    make_image(src / "IMG_4501.HEIC", "people")
    make_image(src / "IMG_4502.heif", "interior")
    make_image(src / "IMG_4503.jpg", "food_drinks")
    return src


def test_heic_and_heif_are_supported_extensions() -> None:
    assert {".heic", ".heif"} <= IMAGE_EXTENSIONS


def test_heic_photos_are_sorted(heic_photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    before = tree_hashes(heic_photos)
    result = sort_folder(heic_photos, tmp_path / "albums", classifier, progress=False)
    copied = {item.original_path.name: item.album for item in result.sorted_files}
    assert copied == {"IMG_4501.HEIC": "People", "IMG_4502.heif": "Interior", "IMG_4503.jpg": "Food & Drinks"}
    assert result.skipped == []
    assert (tmp_path / "albums" / "People" / "IMG_4501.HEIC").read_bytes() == (heic_photos / "IMG_4501.HEIC").read_bytes()
    assert tree_hashes(heic_photos) == before


def test_heic_without_pillow_heif_is_skipped_with_a_clear_reason(heic_photos: Path, tmp_path: Path,
                                                                  classifier: PhotoClassifier,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cafe_sorter.features, "HEIF_AVAILABLE", False)
    result = sort_folder(heic_photos, tmp_path / "albums", classifier, progress=False)
    assert [item.original_path.name for item in result.sorted_files] == ["IMG_4503.jpg"]
    reasons = {path.name: why for path, why in result.skipped}
    assert set(reasons) == {"IMG_4501.HEIC", "IMG_4502.heif"}
    assert all("pillow-heif" in why for why in reasons.values())
