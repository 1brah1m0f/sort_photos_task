"""File-system helpers for the sorter: discovery, validation and non-destructive copying."""
import shutil
from pathlib import Path
from typing import List, Tuple

from .config import IMAGE_EXTENSIONS


def is_supported_image(path: Path) -> bool:
    """Return True if ``path`` has a supported image extension (case-insensitive)."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def list_files(input_dir: Path, recursive: bool) -> Tuple[List[Path], List[Path]]:
    """Return ``(images, other_files)`` found in ``input_dir``, each sorted for reproducibility."""
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = sorted(p for p in candidates if p.is_file())
    images = [p for p in files if is_supported_image(p)]
    others = [p for p in files if not is_supported_image(p)]
    return images, others


def validate_sort_dirs(input_dir: Path, output_dir: Path) -> Tuple[Path, Path]:
    """Resolve both folders and reject unsafe combinations.

    Raises
    ------
    ValueError
        If the input folder does not exist or the output folder is the input folder or inside it.
    """
    source = input_dir.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Input folder does not exist or is not a folder: {input_dir}")
    if target == source or target.is_relative_to(source):
        raise ValueError(f"Output folder {output_dir} must not be the input folder or inside it.")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Output path exists and is not a folder: {output_dir}")
    return source, target


def unique_destination(directory: Path, filename: str) -> Path:
    """Return ``directory/filename``, or ``name_1.ext``, ``name_2.ext``... if that is already taken."""
    candidate = directory / filename
    stem, suffix = Path(filename).stem, Path(filename).suffix
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def copy_into(source: Path, directory: Path) -> Path:
    """Copy ``source`` (with metadata) into ``directory`` without overwriting anything.

    The original file is only read, never moved, renamed or modified.
    """
    destination = unique_destination(directory, source.name)
    shutil.copy2(source, destination)
    return destination
