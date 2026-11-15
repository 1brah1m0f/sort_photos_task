"""Duplicate and near-duplicate detection between an evaluation set and the training/test data.

Three independent signals, all computed from file bytes or pixels only:

* identical SHA256 of the file bytes (exact copies);
* 64-bit difference hash (dHash) of the upright grayscale image, compared by Hamming distance
  (re-encodes, resizes, small crops);
* cosine similarity of the frozen backbone embeddings (the same view of the image the classifier
  gets; catches recompressed or lightly edited copies the hash misses).
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .features import Embedder, ImageLoadError, embed_paths, open_image
from .fileops import list_files

DHASH_MAX_DISTANCE = 6
COSINE_THRESHOLD = 0.95


@dataclass(frozen=True)
class DuplicatePair:
    """A candidate image that matches a reference image by at least one signal."""

    candidate: Path
    reference: Path
    sha256_equal: bool
    dhash_distance: int
    cosine: float


def sha256_of(path: Path) -> str:
    """SHA256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(image: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: compares horizontally adjacent pixels of a (size+1)x size thumbnail."""
    pixels = np.asarray(image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _fingerprints(paths: Sequence[Path]) -> Tuple[List[Path], List[str], List[int]]:
    kept, digests, hashes = [], [], []
    for path in paths:
        try:
            hashes.append(dhash(open_image(path)))
        except ImageLoadError:
            continue
        kept.append(path)
        digests.append(sha256_of(path))
    return kept, digests, hashes


def find_duplicates(candidate_dir: Path, reference_dirs: Sequence[Path], embedder: Embedder, batch_size: int,
                    cosine_threshold: float = COSINE_THRESHOLD, dhash_max_distance: int = DHASH_MAX_DISTANCE,
                    progress: bool = True) -> Tuple[List[DuplicatePair], Dict[str, Any]]:
    """Compare every image under ``candidate_dir`` with every image under ``reference_dirs`` (recursive).

    Returns
    -------
    pairs, summary
        Matching pairs (highest cosine first), and a summary with the number of images compared,
        the closest reference pair overall and the minimum dHash distance seen.
    """
    candidates, _ = list_files(candidate_dir, recursive=True)
    references = [p for d in reference_dirs for p in list_files(d, recursive=True)[0]]
    cand_paths, cand_sha, cand_hash = _fingerprints(candidates)
    ref_paths, ref_sha, ref_hash = _fingerprints(references)
    if not cand_paths or not ref_paths:
        raise ValueError("Need at least one readable candidate image and one readable reference image.")
    if batch_size < 1:
        raise ValueError(f"batch size must be a positive integer, got {batch_size}")

    cand_emb, cand_emb_paths, _ = embed_paths(cand_paths, embedder, batch_size, progress, desc="Embedding candidates")
    ref_emb, ref_emb_paths, _ = embed_paths(ref_paths, embedder, batch_size, progress, desc="Embedding references")
    cand_index = {p: i for i, p in enumerate(cand_emb_paths)}
    ref_index = {p: i for i, p in enumerate(ref_emb_paths)}
    cosine = cand_emb @ ref_emb.T  # embeddings are L2-normalised
    # an image is never its own duplicate (lets a folder be checked against itself)
    same_file = np.array([[c.resolve() == r.resolve() for r in ref_emb_paths] for c in cand_emb_paths])
    cosine = np.where(same_file, -np.inf, cosine)

    pairs: List[DuplicatePair] = []
    distances: List[int] = []
    for i, cand in enumerate(cand_paths):
        for j, ref in enumerate(ref_paths):
            if cand.resolve() == ref.resolve():
                continue
            distance = bin(cand_hash[i] ^ ref_hash[j]).count("1")
            distances.append(distance)
            cos = float(cosine[cand_index[cand], ref_index[ref]])
            same = cand_sha[i] == ref_sha[j]
            if same or distance <= dhash_max_distance or cos >= cosine_threshold:
                pairs.append(DuplicatePair(cand, ref, same, distance, cos))
    pairs.sort(key=lambda p: p.cosine, reverse=True)
    i, j = np.unravel_index(int(cosine.argmax()), cosine.shape)
    summary = {
        "candidates": len(cand_paths),
        "references": len(ref_paths),
        "max_cosine": float(cosine[i, j]),
        "max_cosine_pair": (cand_emb_paths[i], ref_emb_paths[j]),
        "min_dhash_distance": min(distances) if distances else None,
    }
    return pairs, summary

