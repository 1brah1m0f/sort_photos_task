"""Pretrained CLIP backbone: download the weights once, then load the image encoder strictly from local files."""
import hashlib
import os
import urllib.request
import warnings
from pathlib import Path
from typing import Callable, List, Tuple, Union

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import VisionTransformer, build_model

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


__all__ = ["available_models", "checkpoint_path", "download", "expected_sha256", "load"]

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def available_models() -> List[str]:
    """Return the names of the supported CLIP image encoders."""
    return list(_MODELS.keys())


def _model_url(name: str) -> str:
    if name not in _MODELS:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
    return _MODELS[name]


def expected_sha256(name: str) -> str:
    """Return the SHA256 checksum the published checkpoint for ``name`` must have."""
    return _model_url(name).split("/")[-2]


def checkpoint_path(name: str, root: Union[str, Path]) -> Path:
    """Return where the checkpoint for ``name`` lives inside ``root``."""
    return Path(root) / os.path.basename(_model_url(name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, root: Union[str, Path]) -> Path:
    """Download the checkpoint for ``name`` into ``root``. This is the only function that uses the network.

    An existing file with a matching SHA256 checksum is reused without downloading.

    Returns
    -------
    The path of the verified checkpoint file.
    """
    url = _model_url(name)
    os.makedirs(root, exist_ok=True)
    download_target = checkpoint_path(name, root)

    if download_target.exists() and not download_target.is_file():
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if download_target.is_file():
        if _sha256(download_target) == expected_sha256(name):
            return download_target
        print(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    partial = download_target.with_suffix(".part")
    with urllib.request.urlopen(url) as source, open(partial, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if _sha256(partial) != expected_sha256(name):
        partial.unlink()
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")
    partial.replace(download_target)

    return download_target


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _transform(n_px: int) -> Callable[[Image.Image], torch.Tensor]:
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def load(name: str, device: Union[str, torch.device] = "cpu", download_root: Union[str, Path, None] = None) -> Tuple[VisionTransformer, Callable[[Image.Image], torch.Tensor]]:
    """Load a frozen CLIP image encoder from local files only; this never touches the network.

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        directory that ``download()`` saved the checkpoint into

    Returns
    -------
    model : VisionTransformer
        The CLIP image encoder (float32) in eval mode with gradients disabled

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        if download_root is None:
            raise ValueError("download_root is required when loading a model by name")
        model_path = checkpoint_path(name, download_root)
        if not model_path.is_file():
            raise FileNotFoundError(f"Backbone weights not found at {model_path}. Run `photo-sorter download-model` once first.")
    elif os.path.isfile(name):
        model_path = Path(name)
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # the published checkpoints are TorchScript archives; only their weights are used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.jit.load.*", category=FutureWarning)
            state_dict = torch.jit.load(str(model_path), map_location="cpu").state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict).float().to(device)
    model.eval().requires_grad_(False)
    return model, _transform(model.input_resolution)
