"""CLIP Vision Transformer image encoder (architecture from OpenAI CLIP, MIT license).

Only the image tower is kept: the sorter uses CLIP as a frozen feature extractor, so the text
encoder of the original model is never built and its weights are dropped when loading.
"""
from collections import OrderedDict
from typing import Dict

import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """The GELU approximation CLIP was trained with."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Pre-norm transformer block: self-attention followed by an MLP."""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """A stack of residual attention blocks."""

    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """CLIP ViT image encoder: patches -> transformer -> projected [CLS] embedding."""

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        return x @ self.proj

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`forward`, matching the original CLIP API."""
        return self(image)


def build_model(state_dict: Dict[str, torch.Tensor]) -> VisionTransformer:
    """Build the image encoder from a full CLIP ViT checkpoint, ignoring the text-tower weights."""
    if "visual.proj" not in state_dict:
        raise ValueError("Only CLIP Vision Transformer checkpoints (ViT-*) are supported")
    visual = {k[len("visual."):]: v for k, v in state_dict.items() if k.startswith("visual.")}

    width = visual["conv1.weight"].shape[0]
    layers = len([k for k in visual if k.endswith(".attn.in_proj_weight")])
    patch_size = visual["conv1.weight"].shape[-1]
    grid_size = round((visual["positional_embedding"].shape[0] - 1) ** 0.5)

    model = VisionTransformer(
        input_resolution=patch_size * grid_size,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=width // 64,
        output_dim=visual["proj"].shape[1],
    )
    model.load_state_dict(visual)
    return model.eval()
