"""Re-running `sort` into an output folder from a previous run: refused by default, extended with --append."""
import csv
from pathlib import Path

import pytest

from clip.config import PREDICTIONS_FILE
from clip.predict import PhotoClassifier
from clip.sorter import sort_folder
from conftest import make_image, tree_hashes


def _album_files(out: Path) -> list:
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != PREDICTIONS_FILE)


def _csv(out: Path):
    with open(out / PREDICTIONS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, list(reader)


def test_rerun_without_append_is_refused_before_anything_changes(photos: Path, tmp_path: Path, cli_with_classifier,
                                                                 capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "out"
    assert cli_with_classifier(["sort", "--input", str(photos), "--output", str(out), "--recursive"]) == 0
    capsys.readouterr()
    before = tree_hashes(out)

    code = cli_with_classifier(["sort", "--input", str(photos), "--output", str(out), "--recursive"])
    err = capsys.readouterr().err
    assert code == 2
    assert "already contains a previous sort" in err and "--append" in err
    assert tree_hashes(out) == before


@pytest.mark.parametrize("leftover", ["album_file", "csv_only"])
def test_any_previous_output_triggers_the_refusal(leftover: str, photos: Path, tmp_path: Path,
                                                  classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    if leftover == "album_file":
        make_image(out / "Interior" / "old.jpg", "interior")
    else:
        out.mkdir()
        (out / PREDICTIONS_FILE).write_text("original_path,album\n")
    with pytest.raises(ValueError, match="--append"):
        sort_folder(photos, out, classifier, progress=False)


def test_empty_album_folders_are_not_a_previous_run(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    (out / "People").mkdir(parents=True)
    result = sort_folder(photos, out, classifier, recursive=True, progress=False)
    assert len(result.sorted_files) == 4


def test_append_keeps_old_files_adds_new_and_extends_csv(photos: Path, tmp_path: Path, cli_with_classifier,
                                                         capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "out"
    assert cli_with_classifier(["sort", "--input", str(photos), "--output", str(out), "--recursive"]) == 0
    first_files = {rel: (out / rel).read_bytes() for rel in _album_files(out)}
    header1, rows1 = _csv(out)

    more = tmp_path / "more"
    make_image(more / "a1b2.jpg", "food_drinks")      # same name as an earlier copy -> a1b2_1.jpg
    make_image(more / "new_room.png", "interior")
    originals = tree_hashes(photos) | {f"more/{k}": v for k, v in tree_hashes(more).items()}

    code = cli_with_classifier(["sort", "--input", str(more), "--output", str(out), "--append"])
    assert code == 0
    assert "(4 earlier + 2 new rows)" in capsys.readouterr().out

    for rel, data in first_files.items():  # nothing from the first run was replaced
        assert (out / rel).read_bytes() == data
    assert _album_files(out) == sorted(list(first_files) + ["Food & Drinks/a1b2_1.jpg", "Interior/new_room.png"])

    header2, rows2 = _csv(out)
    assert header2 == header1                       # one header, not repeated
    assert rows2[:4] == rows1                       # old rows kept in order
    assert len(rows2) == len(_album_files(out)) == 6  # one row per copy, across both runs
    copied_to = {Path(row[header2.index("copied_to")]).name for row in rows2[4:]}
    assert copied_to == {"a1b2_1.jpg", "new_room.png"}
    assert tree_hashes(photos) | {f"more/{k}": v for k, v in tree_hashes(more).items()} == originals
    assert not list(out.glob(".predictions.csv.*"))  # no temporary file left behind


def test_append_to_a_fresh_folder_writes_header_once(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    sort_folder(photos, out, classifier, recursive=True, progress=False, append=True)
    header, rows = _csv(out)
    assert header[0] == "original_path" and len(rows) == 4


def test_append_refuses_csv_from_a_model_with_other_classes(photos: Path, tmp_path: Path,
                                                            classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / PREDICTIONS_FILE).write_text("original_path,album,p_dogs\nx.jpg,Dogs,0.9\n")
    with pytest.raises(ValueError, match="p_dogs"):
        sort_folder(photos, out, classifier, progress=False, append=True)
    assert not (out / "People").exists()
