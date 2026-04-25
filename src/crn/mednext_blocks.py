"""MedNeXt building blocks (Roy et al., MICCAI 2023; arXiv:2303.09975).

Faithful reimplementation of the MedNeXt inverted-bottleneck block and its
strided (DownBlock) variant. The block is defined as:

    x -> DWConv(kernel=k, groups=C_in) -> GroupNorm
      -> Conv1x1(C_in -> C_in * R)     -> GELU
      -> Conv1x1(C_in * R -> C_out)    -> (+ residual if shapes match)

The DownBlock variant uses stride=2 on the depthwise convolution and a
1x1 strided residual projection so that the skip matches spatially.

Both 2D and 3D variants are provided. The encoder-side configuration
presets for S/B/M/L match Table 1 of Roy et al. 2023.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


Dim = Literal[2, 3]


def _conv_cls(dim: Dim) -> type[nn.Module]:
    return nn.Conv3d if dim == 3 else nn.Conv2d


def _group_norm(dim: Dim, num_channels: int, num_groups: int | None = None) -> nn.Module:
    groups = num_groups if num_groups is not None else min(num_channels, 32)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=max(groups, 1), num_channels=num_channels)


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_ratio: int,
        kernel_size: int = 5,
        dim: Dim = 3,
    ) -> None:
        super().__init__()
        conv_cls = _conv_cls(dim)
        padding = kernel_size // 2
        hidden = in_channels * exp_ratio

        self.dwconv = conv_cls(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.norm = _group_norm(dim, in_channels)
        self.expand = conv_cls(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.compress = conv_cls(hidden, out_channels, kernel_size=1)

        self.residual = in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.expand(h)
        h = self.act(h)
        h = self.compress(h)
        if self.residual:
            h = h + x
        return h


class MedNeXtDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_ratio: int,
        kernel_size: int = 5,
        dim: Dim = 3,
    ) -> None:
        super().__init__()
        conv_cls = _conv_cls(dim)
        padding = kernel_size // 2
        hidden = in_channels * exp_ratio

        self.dwconv = conv_cls(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            groups=in_channels,
        )
        self.norm = _group_norm(dim, in_channels)
        self.expand = conv_cls(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.compress = conv_cls(hidden, out_channels, kernel_size=1)
        self.skip = conv_cls(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.expand(h)
        h = self.act(h)
        h = self.compress(h)
        return h + self.skip(x)


MEDNEXT_VARIANTS: dict[str, dict] = {
    "S": {
        "n_channels": 32,
        "blocks_per_stage": [2, 2, 2, 2],
        "exp_ratios": [2, 3, 4, 4],
        "bottleneck_blocks": 2,
        "bottleneck_exp": 4,
    },
    "B": {
        "n_channels": 32,
        "blocks_per_stage": [2, 2, 2, 2],
        "exp_ratios": [2, 3, 4, 4],
        "bottleneck_blocks": 2,
        "bottleneck_exp": 4,
    },
    "M": {
        "n_channels": 32,
        "blocks_per_stage": [3, 4, 4, 4],
        "exp_ratios": [2, 3, 4, 4],
        "bottleneck_blocks": 4,
        "bottleneck_exp": 4,
    },
    "L": {
        "n_channels": 32,
        "blocks_per_stage": [3, 4, 8, 8],
        "exp_ratios": [3, 4, 8, 8],
        "bottleneck_blocks": 8,
        "bottleneck_exp": 8,
    },
}


def mednext_config(variant: str) -> dict:
    key = variant.upper()
    if key not in MEDNEXT_VARIANTS:
        raise ValueError(f"Unknown MedNeXt variant {variant!r}; choose from {list(MEDNEXT_VARIANTS)}")
    return MEDNEXT_VARIANTS[key]


__all__ = [
    "MedNeXtBlock",
    "MedNeXtDownBlock",
    "MEDNEXT_VARIANTS",
    "mednext_config",
]
