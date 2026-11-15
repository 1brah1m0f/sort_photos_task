"""File-system helpers for the sorter: discovery, validation and non-destructive copying.

Safety rules (the originals must never be modified):

* the output folder may not be the input folder, inside it, or contain it;
* nothing inside the output folder that the sorter writes to (album folders, ``predictions.csv``)
  may be a symlink, so writes cannot be redirected onto an original;
* new files are reserved with an exclusive create (``O_EXCL``), which fails on *any* existing
  path including dangling symlinks, so an existing link is never written through;
* the CSV is written to a temporary file in the output folder and atomically renamed into place.
"""
import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

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
        If the input folder does not exist, or the output folder is the input folder, inside it,
        or contains it.
    """
    source = input_dir.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Input folder does not exist or is not a folder: {input_dir}")
    if target == source or target.is_relative_to(source):
        raise ValueError(f"Output folder {output_dir} must not be the input folder or inside it.")
    if source.is_relative_to(target):
        raise ValueError(f"Output folder {output_dir} must not contain the input folder.")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Output path exists and is not a folder: {output_dir}")
    return source, target


def validate_output_entries(output_dir: Path, album_names: Iterable[str], file_names: Iterable[str]) -> None:
    """Check the paths the sorter will write inside an (already resolved) ``output_dir``.

    Album folders must be absent or real directories; output files must be absent or regular
    files. Symlinks are rejected in both cases, including dangling ones.

    Raises
    ------
    ValueError
        On the first unsafe path, before anything has been created or copied.
    """
    for name in album_names:
        path = output_dir / name
        if path.is_symlink():
            raise ValueError(f"Album folder {path} is a symlink; refusing to write through it.")
        if path.exists() and not path.is_dir():
            raise ValueError(f"Album path {path} exists and is not a folder.")
    for name in file_names:
        path = output_dir / name
        if path.is_symlink():
            raise ValueError(f"Output file {path} is a symlink; refusing to overwrite its target.")
        if path.exists() and not path.is_file():
            raise ValueError(f"Output path {path} exists and is not a regular file.")


def _reserve(path: Path) -> bool:
    """Atomically create an empty ``path``; False if anything (even a dangling symlink) is there."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def _candidates(directory: Path, filename: str) -> Iterable[Path]:
    stem, suffix = Path(filename).stem, Path(filename).suffix
    yield directory / filename
    counter = 1
    while True:
        yield directory / f"{stem}_{counter}{suffix}"
        counter += 1


def unique_destination(directory: Path, filename: str) -> Path:
    """Return ``directory/filename``, or ``name_1.ext``, ``name_2.ext``... if that is already taken.

    Uses ``os.path.lexists`` so a dangling symlink counts as taken.
    """
    return next(c for c in _candidates(directory, filename) if not os.path.lexists(c))


def copy_into(source: Path, directory: Path) -> Path:
    """Copy ``source`` (with metadata) into ``directory`` without overwriting anything.

    The destination name is reserved with an exclusive create before ``shutil.copy2`` fills it,
    so an existing file or symlink is never replaced or followed. The original file is only read.
    """
    for candidate in _candidates(directory, source.name):
        if _reserve(candidate):
            shutil.copy2(source, candidate)
            return candidate
    raise AssertionError("unreachable")


def write_csv_atomic(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> Path:
    """Write a CSV via a temporary file in the same folder, then atomically rename it to ``path``.

    ``os.replace`` replaces a directory entry and never follows a symlink at ``path``.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path
