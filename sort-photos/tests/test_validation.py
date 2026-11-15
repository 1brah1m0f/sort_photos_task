"""Regression tests: invalid arguments and incomplete data fail clearly instead of silently succeeding."""
from pathlib import Path

import pytest

import cafe_sorter.train
from conftest import COLOURS, MeanColourEmbedder, make_image
from cafe_sorter.cli import build_parser, main
from cafe_sorter.config import CLASS_NAMES, CLASSIFIER_FILE, METADATA_FILE
from cafe_sorter.evaluate import evaluate
from cafe_sorter.predict import PhotoClassifier
from cafe_sorter.sorter import sort_folder


def make_dataset(root: Path, per_class: int, corrupt_classes: tuple = ()) -> Path:
    """``root/<class>/`` with ``per_class`` images each; classes in ``corrupt_classes`` get only corrupt JPEGs."""
    for name in CLASS_NAMES:
        for i in range(per_class):
            path = root / name / f"{i}.jpg"
            if name in corrupt_classes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\xff\xd8\xff not really a jpeg")
            else:
                make_image(path, name)
    return root


@pytest.fixture
def fake_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``train()`` use the colour embedder instead of loading CLIP weights."""
    monkeypatch.setattr(cafe_sorter.train, "ClipEmbedder", lambda *args, **kwargs: MeanColourEmbedder())
    monkeypatch.setattr(cafe_sorter.train, "checkpoint_path", lambda name, root: Path(root) / "fake.pt")
    monkeypatch.setattr(cafe_sorter.train, "expected_sha256", lambda name: "0" * 64)


@pytest.mark.parametrize("command", [
    ["sort", "--input", "in", "--output", "out"],
    ["train"],
    ["evaluate"],
])
@pytest.mark.parametrize("value", ["-1", "0", "abc"])
def test_non_positive_batch_size_rejected_by_cli(command: list, value: str, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(command + ["--batch-size", value])
    assert exc.value.code == 2
    assert "--batch-size" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["train", "--folds", "1"], ["sort", "--input", "a", "--output", "b", "--min-confidence", "1.5"]])
def test_other_invalid_arguments_rejected(args: list) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(args)
    assert exc.value.code == 2


def test_negative_batch_size_rejected_by_sort_folder(photos: Path, tmp_path: Path, classifier: PhotoClassifier) -> None:
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="batch size"):
        sort_folder(photos, out, classifier, batch_size=-1, progress=False)
    assert not out.exists()


def test_train_fails_when_a_class_has_zero_readable_images(tmp_path: Path, fake_backbone: None,
                                                           capsys: pytest.CaptureFixture) -> None:
    data = make_dataset(tmp_path / "train", per_class=6, corrupt_classes=("interior",))
    out = tmp_path / "models"
    code = main(["train", "--data", str(data), "--out", str(out), "--backbone-dir", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "interior: 0" in err and "No model was written" in err
    assert not (out / CLASSIFIER_FILE).exists() and not (out / METADATA_FILE).exists()


def test_train_fails_when_readable_images_are_fewer_than_folds(tmp_path: Path, fake_backbone: None) -> None:
    data = make_dataset(tmp_path / "train", per_class=6)
    for i in range(3):  # people: 6 files, only 3 decodable
        (data / "people" / f"{i}.jpg").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="people: 3"):
        cafe_sorter.train.train(data, tmp_path / "models", "ViT-B/32", tmp_path, "cpu", batch_size=4, folds=5)
    assert not (tmp_path / "models").exists()


def test_train_succeeds_with_enough_readable_images(tmp_path: Path, fake_backbone: None) -> None:
    data = make_dataset(tmp_path / "train", per_class=6)
    report = cafe_sorter.train.train(data, tmp_path / "models", "ViT-B/32", tmp_path, "cpu", batch_size=4, folds=3)
    assert report.train_counts == {name: 6 for name in CLASS_NAMES}
    assert (tmp_path / "models" / CLASSIFIER_FILE).is_file()


def test_evaluate_with_zero_readable_images_fails(tmp_path: Path, classifier: PhotoClassifier) -> None:
    data = make_dataset(tmp_path / "test", per_class=2, corrupt_classes=tuple(CLASS_NAMES))
    with pytest.raises(ValueError, match="No readable test images"):
        evaluate(classifier, data, batch_size=8, progress=False)


def test_colour_stand_in_matches_classes() -> None:
    assert set(COLOURS) == set(CLASS_NAMES)


def test_duplicate_check_finds_copies_and_recolours_but_not_other_images(tmp_path: Path) -> None:
    from cafe_sorter.duplicates import find_duplicates

    ref = make_image(tmp_path / "train" / "people" / "a.jpg", "people", size=(64, 48))
    make_image(tmp_path / "train" / "interior" / "b.jpg", "interior", size=(64, 48))
    cand_dir = tmp_path / "holdout"
    (cand_dir / "people").mkdir(parents=True)
    (cand_dir / "people" / "copy.jpg").write_bytes(ref.read_bytes())
    make_image(cand_dir / "food_drinks" / "c.jpg", "food_drinks", size=(64, 48))
    pairs, summary = find_duplicates(cand_dir, [tmp_path / "train"], MeanColourEmbedder(), batch_size=4,
                                     cosine_threshold=0.999, dhash_max_distance=-1, progress=False)
    assert [(p.candidate.name, p.reference.name, p.sha256_equal) for p in pairs] == [("copy.jpg", "a.jpg", True)]
    assert summary["candidates"] == 2 and summary["references"] == 2
    # checking a folder against itself never reports an image as its own duplicate
    pairs, _ = find_duplicates(tmp_path / "train", [tmp_path / "train"], MeanColourEmbedder(), batch_size=4,
                               cosine_threshold=0.999, dhash_max_distance=-1, progress=False)
    assert pairs == []


def test_grouped_cv_keeps_each_group_in_one_fold(tmp_path: Path, fake_backbone: None) -> None:
    import csv

    data = make_dataset(tmp_path / "data" / "train", per_class=6)
    with open(tmp_path / "data" / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "filename", "group"])
        for name in CLASS_NAMES:
            for i in range(6):
                writer.writerow([name, f"{i}.jpg", f"venue{i % 3}"])  # 3 venues, each with every class
    code = main(["train", "--data", str(data), "--out", str(tmp_path / "m"), "--backbone-dir", str(tmp_path),
                 "--folds", "3"])
    assert code == 0
    import json
    training = json.loads((tmp_path / "m" / METADATA_FILE).read_text())["training"]
    assert training["cv_grouped"] and training["n_groups"] == 3 and training["cv_scoring"] == "balanced_accuracy"


def test_groups_file_must_cover_every_image(tmp_path: Path, fake_backbone: None) -> None:
    data = make_dataset(tmp_path / "train", per_class=4)
    groups = tmp_path / "groups.csv"
    groups.write_text("class,filename,group\npeople,0.jpg,a\n")
    with pytest.raises(ValueError, match="have no group"):
        cafe_sorter.train.train(data, tmp_path / "m", "ViT-B/32", tmp_path, "cpu", batch_size=4, folds=2, groups_file=groups)
    assert not (tmp_path / "m").exists()


# --- training labels and groups resolved from the class folder -------------------------------------------------------

def _write_manifest(path: Path, rows: list) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "filename", "group"])
        writer.writerows(rows)
    return path


def _nested_dataset(root: Path) -> Path:
    """Every class has two sessions that reuse the same file names: <class>/session{0,1}/{0,1,2}.jpg"""
    for name in CLASS_NAMES:
        for session in range(2):
            for i in range(3):
                make_image(root / name / f"session{session}" / f"{i}.jpg", name)
    return root


def test_cli_train_nested_layout_with_class_relative_filenames(tmp_path: Path, fake_backbone: None) -> None:
    import json

    data = _nested_dataset(tmp_path / "data" / "train")
    rows = [[name, f"session{s}/{i}.jpg", f"venue{s}-{i % 2}"] for name in CLASS_NAMES for s in range(2) for i in range(3)]
    _write_manifest(tmp_path / "data" / "manifest.csv", rows)
    out = tmp_path / "models"
    code = main(["train", "--data", str(data), "--out", str(out), "--backbone-dir", str(tmp_path), "--folds", "2"])
    assert code == 0
    training = json.loads((out / METADATA_FILE).read_text())["training"]
    # labels come from the class folder, not from the 'session' sub-folder
    assert training["train_counts"] == {name: 6 for name in CLASS_NAMES}
    assert training["cv_grouped"] and training["n_groups"] == 4


def test_load_groups_maps_nested_photo_to_its_class(tmp_path: Path) -> None:
    photo = make_image(tmp_path / "train" / "people" / "session" / "photo.jpg", "people")
    manifest = _write_manifest(tmp_path / "m.csv", [["people", "photo.jpg", "cafe-a"]])
    assert cafe_sorter.train.load_groups(manifest, [photo], tmp_path / "train") == ["cafe-a"]
    assert cafe_sorter.train.class_and_relpath(photo, tmp_path / "train") == ("people", "session/photo.jpg")


def test_cli_train_rejects_bare_filename_shared_by_nested_photos(tmp_path: Path, fake_backbone: None,
                                                                 capsys: pytest.CaptureFixture) -> None:
    data = _nested_dataset(tmp_path / "data" / "train")
    rows = [[name, f"{i}.jpg", f"venue{i}"] for name in CLASS_NAMES for i in range(3)]  # bare names, ambiguous
    _write_manifest(tmp_path / "data" / "manifest.csv", rows)
    out = tmp_path / "models"
    code = main(["train", "--data", str(data), "--out", str(out), "--backbone-dir", str(tmp_path), "--folds", "2"])
    assert code == 2
    err = capsys.readouterr().err
    assert "share a file name inside their class" in err and "session0/0.jpg" in err
    assert not out.exists()


def test_cli_train_rejects_conflicting_manifest_rows(tmp_path: Path, fake_backbone: None,
                                                     capsys: pytest.CaptureFixture) -> None:
    data = make_dataset(tmp_path / "data" / "train", per_class=4)
    rows = [[name, f"{i}.jpg", f"g{i}"] for name in CLASS_NAMES for i in range(4)] + [["people", "0.jpg", "other"]]
    _write_manifest(tmp_path / "data" / "manifest.csv", rows)
    code = main(["train", "--data", str(data), "--out", str(tmp_path / "m"), "--backbone-dir", str(tmp_path), "--folds", "2"])
    assert code == 2
    assert "Ambiguous rows" in capsys.readouterr().err


def test_cli_train_flat_layout_with_bare_filenames(tmp_path: Path, fake_backbone: None) -> None:
    import json

    data = make_dataset(tmp_path / "data" / "train", per_class=4)
    _write_manifest(tmp_path / "data" / "manifest.csv",
                    [[name, f"{i}.jpg", f"cafe{i % 2}"] for name in CLASS_NAMES for i in range(4)])
    code = main(["train", "--data", str(data), "--out", str(tmp_path / "m"), "--backbone-dir", str(tmp_path), "--folds", "2"])
    assert code == 0
    training = json.loads((tmp_path / "m" / METADATA_FILE).read_text())["training"]
    assert training["n_groups"] == 2 and training["train_counts"] == {name: 4 for name in CLASS_NAMES}
