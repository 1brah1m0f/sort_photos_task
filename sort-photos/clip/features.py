"""Image loading and frozen-backbone feature extraction.

Only pixel content is used: filenames, folder names and EXIF text fields are never passed
to the model. The EXIF *orientation* tag is applied so rotated phone photos look upright.
"""
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from .clip import load as load_backbone
from .config import default_device

logger = logging.getLogger(__name__)

Skipped = Tuple[Path, str]


class ImageLoadError(Exception):
    """Raised when a file cannot be decoded as an image."""


def open_image(path: Path) -> Image.Image:
    """Decode ``path`` fully, apply EXIF orientation and return an RGB image.

    Raises
    ------
    ImageLoadError
        If the file is missing, corrupt, truncated or not an image.
    """
    try:
        with Image.open(path) as img:
            img.load()
            upright = ImageOps.exif_transpose(img)
            return upright.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(f"{type(exc).__name__}: {exc}") from exc


class Embedder(Protocol):
    """Anything that maps a batch of images to one feature vector per image."""

    name: str

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return an array of shape (len(images), dim)."""
        ...


class ClipEmbedder:
    """Frozen CLIP image encoder producing L2-normalised embeddings."""

    def __init__(self, backbone: str, backbone_dir: Path, device: Optional[str] = None) -> None:
        self.name = backbone
        self.device = device or default_device()
        self.model, self.preprocess = load_backbone(backbone, device=self.device, download_root=backbone_dir)

    @torch.no_grad()
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Encode a batch of RGB images."""
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        features = self.model.encode_image(batch).float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)


def iter_image_batches(paths: Sequence[Path], batch_size: int) -> Iterator[Tuple[List[Path], List[Image.Image], List[Skipped]]]:
    """Yield ``(loaded_paths, images, skipped)`` for consecutive chunks of ``paths``.

    Files that cannot be decoded are logged and reported in ``skipped`` instead of raising.
    """
    for start in range(0, len(paths), batch_size):
        loaded_paths: List[Path] = []
        images: List[Image.Image] = []
        skipped: List[Skipped] = []
        for path in paths[start:start + batch_size]:
            try:
                images.append(open_image(path))
                loaded_paths.append(path)
            except ImageLoadError as exc:
                logger.warning("Skipping unreadable image %s (%s)", path, exc)
                skipped.append((path, f"unreadable image: {exc}"))
        yield loaded_paths, images, skipped


def embed_paths(paths: Sequence[Path], embedder: Embedder, batch_size: int, progress: bool = True,
                desc: str = "Embedding") -> Tuple[np.ndarray, List[Path], List[Skipped]]:
    """Embed every readable image in ``paths``.

    Returns
    -------
    features, embedded_paths, skipped
        ``features[i]`` belongs to ``embedded_paths[i]``.
    """
    chunks: List[np.ndarray] = []
    embedded: List[Path] = []
    skipped: List[Skipped] = []
    with tqdm(total=len(paths), desc=desc, unit="img", disable=not progress) as bar:
        for loaded_paths, images, batch_skipped in iter_image_batches(paths, batch_size):
            if images:
                chunks.append(embedder.embed(images))
                embedded.extend(loaded_paths)
            skipped.extend(batch_skipped)
            bar.update(len(loaded_paths) + len(batch_skipped))
    features = np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
    return features, embedded, skipped
