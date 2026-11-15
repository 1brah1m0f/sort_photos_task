"""Regression tests: the sorter must never write through symlinks onto the original photos."""
import os
from pathlib import Path

import pytest

from conftest import make_image, tree_hashes
from clip.config import PREDICTIONS_FILE
from clip.fileops import copy_into, unique_destination, write_csv_atomic
from clip.predict import PhotoClassifier
from clip.sorter import sort_folder


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
    sort_folder(photos, out, classifier, recursive=True, progress=False)
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
