"""Default paths are absolute and anchored at the project, so the CLI works from any directory."""
from pathlib import Path

import pytest

import cafe_sorter
from cafe_sorter import config
from cafe_sorter.cli import main
from cafe_sorter.predict import PhotoClassifier, load_artifact
from conftest import make_image

PROJECT = Path(cafe_sorter.__file__).resolve().parent.parent


def test_project_root_is_the_package_parent() -> None:
    assert config.PROJECT_ROOT == PROJECT
    assert (config.PROJECT_ROOT / "setup.py").is_file()


@pytest.mark.parametrize("name", ["DEFAULT_TRAIN_DIR", "DEFAULT_TEST_DIR", "DEFAULT_MODEL_DIR",
                                  "DEFAULT_BACKBONE_DIR", "DEFAULT_REPORT_DIR"])
def test_defaults_are_absolute_and_inside_the_project(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = getattr(config, name)
    assert path.is_absolute()
    assert path.is_relative_to(PROJECT)


def test_default_folders_exist_from_another_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for path in (config.DEFAULT_TRAIN_DIR, config.DEFAULT_TEST_DIR, config.DEFAULT_MODEL_DIR, config.DEFAULT_REPORT_DIR):
        assert path.is_dir(), path


def test_shipped_classifier_loads_from_another_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    estimator, metadata = load_artifact(config.DEFAULT_MODEL_DIR)
    assert list(metadata["class_names"]) == list(config.CLASS_NAMES)
    assert hasattr(estimator, "predict_proba")


def test_cli_sort_from_another_directory_uses_package_model_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                  classifier: PhotoClassifier) -> None:
    seen = {}

    def fake_load(model_dir: Path, backbone_dir: Path, device=None) -> PhotoClassifier:
        seen.update(model_dir=model_dir, backbone_dir=backbone_dir)
        return classifier

    monkeypatch.setattr(PhotoClassifier, "load", staticmethod(fake_load))
    workdir = tmp_path / "elsewhere"
    make_image(workdir / "in" / "a.jpg", "people")
    monkeypatch.chdir(workdir)

    assert main(["sort", "--input", "in", "--output", "out"]) == 0
    assert seen == {"model_dir": PROJECT / "models", "backbone_dir": PROJECT / "models" / "backbone"}
    assert (workdir / "out" / "People" / "a.jpg").is_file()


def test_python_m_cafe_sorter_runs_from_another_directory(tmp_path: Path) -> None:
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "cafe_sorter", "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0
    assert done.stdout.startswith("usage: photo-sorter")
