"""--unsure: photos below --min-confidence go to an Unsure album instead of their best guess."""
import csv
import os
from pathlib import Path

import pytest
from PIL import Image

from clip.config import PREDICTIONS_FILE, UNSURE_ALBUM
from clip.predict import PhotoClassifier
from clip.sorter import sort_folder
from conftest import make_image, tree_hashes


@pytest.fixture
def mixed_photos(tmp_path: Path) -> Path:
    """Two clear photos and one red/blue mix that the colour classifier splits ~0.50 / 0.48."""
    src = tmp_path / "photos"
    make_image(src / "latte.jpg", "food_drinks")
    make_image(src / "guests.jpg", "people")
    Image.new("RGB", (24, 16), (125, 30, 125)).save(src / "blurry.png")
    return src


def _rows(out: Path) -> dict:
    with open(out / PREDICTIONS_FILE, newline="", encoding="utf-8") as f:
        return {Path(row["original_path"]).name: row for row in csv.DictReader(f)}


def test_unsure_moves_low_confidence_photos_to_unsure_album(mixed_photos: Path, tmp_path: Path,
                                                            classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    before = tree_hashes(mixed_photos)
    result = sort_folder(mixed_photos, out, classifier, min_confidence=0.8, unsure=True, progress=False)

    assert (out / UNSURE_ALBUM / "blurry.png").is_file()
    assert not (out / "Food & Drinks" / "blurry.png").exists()
    assert (out / "Food & Drinks" / "latte.jpg").is_file() and (out / "People" / "guests.jpg").is_file()
    assert result.counts() == {"Food & Drinks": 1, "People": 1, "Interior": 0, UNSURE_ALBUM: 1}

    rows = _rows(out)
    assert rows["blurry.png"]["album"] == UNSURE_ALBUM
    assert rows["blurry.png"]["predicted_album"] == "Food & Drinks"
    assert rows["blurry.png"]["low_confidence"] == "True"
    probs = [float(rows["blurry.png"][f"p_{c}"]) for c in classifier.class_names]
    assert sum(probs) == pytest.approx(1.0, abs=1e-3) and max(probs) < 0.8
    assert rows["latte.jpg"]["album"] == rows["latte.jpg"]["predicted_album"] == "Food & Drinks"
    assert tree_hashes(mixed_photos) == before


def test_without_unsure_low_confidence_photos_stay_in_best_album(mixed_photos: Path, tmp_path: Path,
                                                                 classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    result = sort_folder(mixed_photos, out, classifier, min_confidence=0.8, progress=False)
    assert not (out / UNSURE_ALBUM).exists()
    assert (out / "Food & Drinks" / "blurry.png").is_file()
    rows = _rows(out)
    assert rows["blurry.png"]["album"] == rows["blurry.png"]["predicted_album"] == "Food & Drinks"
    assert rows["blurry.png"]["low_confidence"] == "True"
    assert [item.original_path.name for item in result.low_confidence] == ["blurry.png"]


def test_cli_unsure_needs_a_threshold(mixed_photos: Path, tmp_path: Path, cli_with_classifier,
                                      capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "out"
    assert cli_with_classifier(["sort", "--input", str(mixed_photos), "--output", str(out), "--unsure"]) == 2
    assert "--min-confidence" in capsys.readouterr().err
    assert not out.exists()


def test_cli_unsure_summary(mixed_photos: Path, tmp_path: Path, cli_with_classifier,
                            capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "out"
    code = cli_with_classifier(["sort", "--input", str(mixed_photos), "--output", str(out),
                                "--min-confidence", "0.8", "--unsure"])
    printed = capsys.readouterr().out
    assert code == 0
    assert f"  {UNSURE_ALBUM}" in printed and "best guess Food & Drinks" in printed


def test_unsure_symlink_is_rejected_even_without_flag(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / UNSURE_ALBUM).symlink_to(photos, target_is_directory=True)
    before = tree_hashes(photos)
    for unsure in (True, False):
        with pytest.raises(ValueError, match="symlink"):
            sort_folder(photos, out, classifier, min_confidence=0.5, unsure=unsure, progress=False)
    assert tree_hashes(photos) == before
    assert sorted(os.listdir(out)) == [UNSURE_ALBUM]
