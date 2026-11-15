"""Regression tests: the sorter must never write through symlinks onto the original photos."""
import os
from pathlib import Path

import pytest

from conftest import make_image, tree_hashes
from cafe_sorter.config import PREDICTIONS_FILE
from cafe_sorter.fileops import copy_into, unique_destination, write_csv_atomic
from cafe_sorter.predict import PhotoClassifier
from cafe_sorter.sorter import sort_folder


def test_csv_symlink_to_input_photo_is_rejected(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / PREDICTIONS_FILE).symlink_to(photos / "a1b2.jpg")
    before = tree_hashes(photos)
    with pytest.raises(ValueError, match="symlink"):
        sort_folder(photos, out, classifier, recursive=True, progress=False)
    assert tree_hashes(photos) == before
    assert [p.name for p in out.iterdir()] == [PREDICTIONS_FILE]  # validated before any album/copy


def test_atomic_csv_write_replaces_link_not_target(tmp_path: Path) -> None:
    victim = make_image(tmp_path / "photos" / "victim.jpg", "people")
    before = victim.read_bytes()
    link = tmp_path / "out" / PREDICTIONS_FILE
    link.parent.mkdir()
    link.symlink_to(victim)
    write_csv_atomic(link, ["a"], [[1]])
    assert victim.read_bytes() == before
    assert not link.is_symlink() and link.read_text() == "a\n1\n"
    assert [p.name for p in link.parent.iterdir()] == [PREDICTIONS_FILE]  # no temp file left behind


@pytest.mark.parametrize("target", ["input", "elsewhere", "dangling"])
def test_album_symlink_is_rejected(target: str, photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    out.mkdir()
    destinations = {"input": photos, "elsewhere": tmp_path / "elsewhere", "dangling": tmp_path / "missing"}
    if target == "elsewhere":
        destinations[target].mkdir()
    (out / "People").symlink_to(destinations[target], target_is_directory=True)
    before = tree_hashes(photos)
    with pytest.raises(ValueError, match="symlink"):
        sort_folder(photos, out, classifier, recursive=True, progress=False)
    assert tree_hashes(photos) == before
    assert not (out / "Food & Drinks").exists()  # nothing created before validation failed


def test_album_path_that_is_a_file_is_rejected(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "Interior").write_text("not a folder")
    with pytest.raises(ValueError, match="not a folder"):
        sort_folder(photos, out, classifier, progress=False)


def test_existing_symlinks_inside_album_are_never_followed(photos: Path, tmp_path: Path,
                                                           classifier: PhotoClassifier) -> None:
    """A link named like an incoming photo, to an original or to nowhere, is treated as a collision."""
    out = tmp_path / "out"
    album = out / "Food & Drinks"
    album.mkdir(parents=True)
    (album / "a1b2.jpg").symlink_to(photos / "IMG_0001.JPEG")  # live link to another original
    victim = tmp_path / "victim.jpg"
    (album / "q.webp").symlink_to(victim)  # dangling link
    before = tree_hashes(photos)
    sort_folder(photos, out, classifier, recursive=True, progress=False, append=True)
    assert tree_hashes(photos) == before
    assert not os.path.lexists(victim)
    assert (album / "a1b2_1.jpg").is_file() and (album / "q_1.webp").is_file()
    assert (album / "a1b2.jpg").is_symlink() and (album / "q.webp").is_symlink()


def test_collision_helpers_treat_dangling_link_as_taken(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").symlink_to(tmp_path / "nowhere.jpg")
    assert unique_destination(tmp_path, "photo.jpg").name == "photo_1.jpg"
    source = make_image(tmp_path / "src" / "photo.jpg", "interior")
    assert copy_into(source, tmp_path).name == "photo_1.jpg"
    assert not (tmp_path / "nowhere.jpg").exists()


def test_input_inside_output_is_rejected(tmp_path: Path, classifier: PhotoClassifier) -> None:
    src = tmp_path / "out" / "People"
    make_image(src / "a.jpg", "people")
    with pytest.raises(ValueError, match="contain the input"):
        sort_folder(src, tmp_path / "out", classifier, progress=False)


def _album_snapshot(album: Path) -> dict:
    return {p.name: p.read_bytes() for p in album.iterdir()}


def test_missing_source_after_reservation_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The source vanishes between reserving the destination and copying it."""
    import cafe_sorter.fileops as fileops

    album = tmp_path / "album"
    existing = make_image(album / "photo.jpg", "people")
    before = _album_snapshot(album)
    source = make_image(tmp_path / "src" / "photo.jpg", "food_drinks")
    real_reserve = fileops._reserve

    def reserve_then_delete_source(path: Path) -> bool:
        reserved = real_reserve(path)
        if reserved:
            source.unlink()
        return reserved

    monkeypatch.setattr(fileops, "_reserve", reserve_then_delete_source)
    with pytest.raises(FileNotFoundError):
        copy_into(source, album)
    assert _album_snapshot(album) == before  # no photo_1.jpg, existing file byte-identical
    assert existing.read_bytes() == before["photo.jpg"]


def test_partial_write_is_removed_and_error_surfaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A copy that writes some bytes and then fails (e.g. disk full) leaves nothing behind."""
    import cafe_sorter.fileops as fileops

    album = tmp_path / "album"
    make_image(album / "photo.jpg", "people")
    before = _album_snapshot(album)
    source = make_image(tmp_path / "src" / "photo.jpg", "interior", size=(200, 200))

    def partial_copy(src: Path, dst: Path) -> None:
        Path(dst).write_bytes(Path(src).read_bytes()[:100])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fileops.shutil, "copy2", partial_copy)
    with pytest.raises(OSError, match="No space left"):
        copy_into(source, album)
    assert _album_snapshot(album) == before


def test_successful_copy_keeps_bytes_and_metadata(tmp_path: Path) -> None:
    source = make_image(tmp_path / "src" / "photo.jpg", "people")
    os.utime(source, (1_000_000_000, 1_000_000_000))
    album = tmp_path / "album"
    album.mkdir()
    copied = copy_into(source, album)
    assert copied.read_bytes() == source.read_bytes()
    assert copied.stat().st_mtime == source.stat().st_mtime


def test_sort_reports_failed_copy_and_continues(photos: Path, tmp_path: Path, classifier: PhotoClassifier,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import cafe_sorter.fileops as fileops

    real_copy2 = fileops.shutil.copy2

    def flaky_copy2(src, dst):
        if Path(src).name == "x.png":
            Path(dst).write_bytes(b"partial")
            raise OSError(5, "Input/output error")
        return real_copy2(src, dst)

    monkeypatch.setattr(fileops.shutil, "copy2", flaky_copy2)
    out = tmp_path / "out"
    result = sort_folder(photos, out, classifier, recursive=True, progress=False)
    assert len(result.sorted_files) == 3
    assert [(p.name, why.split(":")[0]) for p, why in result.skipped] == [("x.png", "copy failed")]
    assert not (out / "Interior" / "x.png").exists()
    for item in result.sorted_files:
        assert item.copied_to.read_bytes() == item.original_path.read_bytes()


# --- exit status of `photo-sorter sort` when copies fail ----------------------------------------------------------

def _failing_copy2(monkeypatch: pytest.MonkeyPatch, fail_names: set) -> None:
    import cafe_sorter.fileops as fileops

    real_copy2 = fileops.shutil.copy2

    def copy2(src, dst):
        if fail_names == {"*"} or Path(src).name in fail_names:
            Path(dst).write_bytes(Path(src).read_bytes()[:10])  # partial write, then failure
            raise OSError(28, "No space left on device")
        return real_copy2(src, dst)

    monkeypatch.setattr(fileops.shutil, "copy2", copy2)


def _album_files(out: Path) -> list:
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != PREDICTIONS_FILE)


def test_cli_sort_exits_1_when_every_copy_fails(photos: Path, tmp_path: Path, cli_with_classifier,
                                                monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _failing_copy2(monkeypatch, {"*"})
    (photos / "corrupt.jpg").write_bytes(b"\xff\xd8\xff not a jpeg")
    before = tree_hashes(photos)
    out = tmp_path / "out"
    code = cli_with_classifier(["sort", "--input", str(photos), "--output", str(out), "--recursive"])
    captured = capsys.readouterr()
    assert code == 1
    assert "Copy failures: 4 of 4" in captured.err and "every copy failed" in captured.err
    assert _album_files(out) == []  # no empty or partial images left behind
    assert tree_hashes(photos) == before


def test_cli_sort_partial_copy_failure_still_succeeds(photos: Path, tmp_path: Path, cli_with_classifier,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture) -> None:
    _failing_copy2(monkeypatch, {"x.png"})
    before = tree_hashes(photos)
    out = tmp_path / "out"
    code = cli_with_classifier(["sort", "--input", str(photos), "--output", str(out), "--recursive"])
    assert code == 0
    assert "Copy failures: 1 of 4" in capsys.readouterr().err
    assert _album_files(out) == ["Food & Drinks/a1b2.jpg", "Food & Drinks/q.webp", "People/IMG_0001.JPEG"]
    for rel in _album_files(out):
        assert (out / rel).read_bytes() == next(photos.rglob(Path(rel).name)).read_bytes()
    assert tree_hashes(photos) == before


def test_cli_sort_only_corrupt_files_is_not_a_copy_failure(tmp_path: Path, cli_with_classifier,
                                                           capsys: pytest.CaptureFixture) -> None:
    src = tmp_path / "photos"
    src.mkdir()
    (src / "broken.jpg").write_bytes(b"garbage")
    (src / "notes.txt").write_text("hi")
    code = cli_with_classifier(["sort", "--input", str(src), "--output", str(tmp_path / "out")])
    assert code == 0
    captured = capsys.readouterr()
    assert "Skipped: 2" in captured.out and "Copy failures" not in captured.err
