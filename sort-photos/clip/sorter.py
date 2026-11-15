"""Sort a folder of photos into album folders by copying each file to its predicted album."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import ALBUMS, PREDICTIONS_FILE
from .features import Skipped, iter_image_batches
from .fileops import copy_into, list_files, validate_output_entries, validate_sort_dirs, write_csv_atomic
from .predict import PhotoClassifier

logger = logging.getLogger(__name__)


@dataclass
class SortedFile:
    """One copied photo."""

    original_path: Path
    copied_to: Path
    album: str
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


def create_albums(output_dir: Path, class_names: List[str]) -> Dict[str, Path]:
    """Create one album folder per class and return ``{album name: folder}``."""
    albums = {}
    for name in class_names:
        album = ALBUMS.get(name, name)
        folder = output_dir / album
        folder.mkdir(parents=True, exist_ok=True)
        albums[album] = folder
    return albums


def write_predictions_csv(result: SortResult, class_names: List[str]) -> Path:
    """Atomically write one row per copied photo to ``predictions.csv`` in the output folder."""
    header = ["original_path", "album", "confidence", "low_confidence", "copied_to"] + [f"p_{name}" for name in class_names]
    rows = (
        [str(item.original_path), item.album, f"{item.confidence:.4f}", item.low_confidence, str(item.copied_to)]
        + [f"{item.probabilities[name]:.4f}" for name in class_names]
        for item in result.sorted_files
    )
    return write_csv_atomic(result.output_dir / PREDICTIONS_FILE, header, rows)


def sort_folder(input_dir: Path, output_dir: Path, classifier: PhotoClassifier, recursive: bool = False,
                min_confidence: float = 0.0, batch_size: int = 32, progress: bool = True) -> SortResult:
    """Classify every photo in ``input_dir`` and copy it into its album under ``output_dir``.

    Originals are never moved, renamed or modified. Files that are not readable images are skipped
    with a warning. Photos below ``min_confidence`` are still copied to their best album but flagged.

    Raises
    ------
    ValueError
        For an invalid batch size or confidence threshold, or unsafe input/output paths. All checks
        run before any folder is created or any file is copied.
    """
    if batch_size < 1:
        raise ValueError(f"batch size must be a positive integer, got {batch_size}")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence}")
    source, target = validate_sort_dirs(input_dir, output_dir)
    validate_output_entries(target, [ALBUMS.get(c, c) for c in classifier.class_names], [PREDICTIONS_FILE])
    images, others = list_files(source, recursive)
    result = SortResult(output_dir=target, albums=create_albums(target, classifier.class_names))
    for path in others:
        logger.warning("Skipping unsupported file %s", path)
        result.skipped.append((path, "unsupported file type"))

    with tqdm(total=len(images), desc="Sorting", unit="img", disable=not progress) as bar:
        for loaded_paths, batch_images, batch_skipped in iter_image_batches(images, batch_size):
            result.skipped.extend(batch_skipped)
            if batch_images:
                probabilities = classifier.predict_images(batch_images)
                for prediction in classifier.to_predictions(loaded_paths, probabilities):
                    album = ALBUMS.get(prediction.label, prediction.label)
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
                        confidence=prediction.confidence,
                        low_confidence=min_confidence > 0 and prediction.confidence < min_confidence,
                        probabilities=prediction.probabilities,
                    ))
            bar.update(len(loaded_paths) + len(batch_skipped))

    write_predictions_csv(result, classifier.class_names)
    return result
