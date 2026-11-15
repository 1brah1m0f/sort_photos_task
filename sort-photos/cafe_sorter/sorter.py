"""Sort a folder of photos into album folders by copying each file to its predicted album."""
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from tqdm import tqdm

from .config import ALBUMS, PREDICTIONS_FILE, UNSURE_ALBUM
from .features import Skipped, iter_image_batches
from .fileops import (copy_into, existing_output, list_files, read_csv_rows, validate_output_entries,
                      validate_sort_dirs, write_csv_atomic)
from .predict import PhotoClassifier

logger = logging.getLogger(__name__)


class NoImagesFound(Exception):
    """The input folder has no files with a supported image extension."""


@dataclass
class SortedFile:
    """One copied photo. ``album`` is where it was copied (``Unsure`` or the best class);
    ``predicted_album`` is always the best class."""

    original_path: Path
    copied_to: Path
    album: str
    predicted_album: str
    confidence: float
    low_confidence: bool
    probabilities: Dict[str, float]


@dataclass
class SortResult:
    """Outcome of :func:`sort_folder`."""

    output_dir: Path
    albums: Dict[str, Path]
    sorted_files: List[SortedFile] = field(default_factory=list)
    skipped: List[Skipped] = field(default_factory=list)
    copy_failures: List[Skipped] = field(default_factory=list)
    previous_rows: int = 0  # rows already in predictions.csv before an --append run

    @property
    def all_copies_failed(self) -> bool:
        """True if at least one copy was attempted and none succeeded."""
        return bool(self.copy_failures) and not self.sorted_files

    def counts(self) -> Dict[str, int]:
        """Number of photos copied into each album."""
        totals = {album: 0 for album in self.albums}
        for item in self.sorted_files:
            totals[item.album] += 1
        return totals

    @property
    def low_confidence(self) -> List[SortedFile]:
        return [item for item in self.sorted_files if item.low_confidence]


def album_names(class_names: Sequence[str], unsure: bool = False) -> List[str]:
    """Album folder names, in class order, plus ``Unsure`` last when requested."""
    return [ALBUMS.get(name, name) for name in class_names] + ([UNSURE_ALBUM] if unsure else [])


def create_albums(output_dir: Path, names: Sequence[str]) -> Dict[str, Path]:
    """Create the album folders and return ``{album name: folder}``."""
    albums = {}
    for album in names:
        folder = output_dir / album
        folder.mkdir(parents=True, exist_ok=True)
        albums[album] = folder
    return albums


def predictions_header(class_names: List[str]) -> List[str]:
    return (["original_path", "album", "predicted_album", "confidence", "low_confidence", "copied_to"]
            + [f"p_{name}" for name in class_names])


def write_predictions_csv(result: SortResult, class_names: List[str], previous_rows: Sequence[Sequence[str]] = ()) -> Path:
    """Atomically write ``predictions.csv``: ``previous_rows`` (from an earlier run) then one row per new copy."""
    new_rows = (
        [str(item.original_path), item.album, item.predicted_album, f"{item.confidence:.4f}", item.low_confidence,
         str(item.copied_to)]
        + [f"{item.probabilities[name]:.4f}" for name in class_names]
        for item in result.sorted_files
    )
    return write_csv_atomic(result.output_dir / PREDICTIONS_FILE, predictions_header(class_names),
                            itertools.chain(previous_rows, new_rows))


def sort_folder(input_dir: Path, output_dir: Path, classifier: PhotoClassifier, recursive: bool = False,
                min_confidence: float = 0.0, batch_size: int = 32, progress: bool = True,
                append: bool = False, unsure: bool = False) -> SortResult:
    """Classify every photo in ``input_dir`` and copy it into its album under ``output_dir``.

    Originals are never moved, renamed or modified. Files that are not readable images are skipped
    with a warning. Photos below ``min_confidence`` are flagged and copied to their best album, or to
    an ``Unsure`` album when ``unsure`` is True.

    If the output already holds a previous run (files in an album folder or a ``predictions.csv``),
    sorting stops unless ``append`` is True. With ``append``, existing files are kept, new copies get
    the usual ``_1``, ``_2`` names on clashes, and the new rows are added to ``predictions.csv``
    (old and new rows are rewritten together through a temporary file and an atomic rename).

    Raises
    ------
    ValueError
        For an invalid batch size or confidence threshold, ``unsure`` without a threshold, unsafe input/output paths, or an output
        folder from a previous run without ``append``. All checks run before any folder is created
        or any file is copied.
    NoImagesFound
        If the input folder has no supported image files; nothing is created in that case.
    """
    if batch_size < 1:
        raise ValueError(f"batch size must be a positive integer, got {batch_size}")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence}")
    if unsure and min_confidence <= 0:
        raise ValueError("--unsure needs --min-confidence above 0 (e.g. --min-confidence 0.6)")
    source, target = validate_sort_dirs(input_dir, output_dir)
    # Always check Unsure too: a link or file there could come from an earlier --unsure run.
    all_albums = album_names(classifier.class_names, unsure=True)
    validate_output_entries(target, all_albums, [PREDICTIONS_FILE])
    images, others = list_files(source, recursive)
    if not images:
        raise NoImagesFound(f"No images found in {input_dir}" + ("" if recursive else " (use --recursive for sub-folders)"))
    previous = existing_output(target, all_albums, [PREDICTIONS_FILE])
    if previous and not append:
        raise ValueError(f"Output folder {output_dir} already contains a previous sort ({len(previous)} entries, "
                         f"e.g. {previous[0]}). Re-run with --append to add to it, or choose an empty folder.")
    csv_path = target / PREDICTIONS_FILE
    previous_rows = read_csv_rows(csv_path, predictions_header(classifier.class_names)) if csv_path.exists() else []
    result = SortResult(output_dir=target, albums=create_albums(target, album_names(classifier.class_names, unsure)),
                        previous_rows=len(previous_rows))
    for path in others:
        logger.warning("Skipping unsupported file %s", path)
        result.skipped.append((path, "unsupported file type"))

    with tqdm(total=len(images), desc="Sorting", unit="img", disable=not progress) as bar:
        for loaded_paths, batch_images, batch_skipped in iter_image_batches(images, batch_size):
            result.skipped.extend(batch_skipped)
            if batch_images:
                probabilities = classifier.predict_images(batch_images)
                for prediction in classifier.to_predictions(loaded_paths, probabilities):
                    predicted = ALBUMS.get(prediction.label, prediction.label)
                    low = min_confidence > 0 and prediction.confidence < min_confidence
                    album = UNSURE_ALBUM if unsure and low else predicted
                    try:
                        copied_to = copy_into(prediction.path, result.albums[album])
                    except OSError as exc:  # copy_into already removed its incomplete output
                        logger.warning("Could not copy %s: %s", prediction.path, exc)
                        failure = (prediction.path, f"copy failed: {type(exc).__name__}: {exc}")
                        result.copy_failures.append(failure)
                        result.skipped.append(failure)
                        continue
                    result.sorted_files.append(SortedFile(
                        original_path=prediction.path,
                        copied_to=copied_to,
                        album=album,
                        predicted_album=predicted,
                        confidence=prediction.confidence,
                        low_confidence=low,
                        probabilities=prediction.probabilities,
                    ))
            bar.update(len(loaded_paths) + len(batch_skipped))

    write_predictions_csv(result, classifier.class_names, previous_rows)
    return result
