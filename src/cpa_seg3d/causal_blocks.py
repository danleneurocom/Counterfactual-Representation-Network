from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class CausalFiLM3D(nn.Module):
    """Feature-wise modulation from disease/context latents."""

    def __init__(self, latent_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.proj = nn.Sequential(
            nn.Linear(int(latent_dim) * 3, int(latent_dim)),
            nn.SiLU(),
            nn.Linear(int(latent_dim), self.channels * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, feature: Tensor, z_d: Tensor, z_c: Tensor, z_dc: Tensor) -> Tensor:
        if z_d.shape != z_c.shape or z_d.shape != z_dc.shape:
            raise ValueError("z_d, z_c, and z_dc must have matching shapes.")
        gamma, beta = self.proj(torch.cat([z_d, z_c, z_dc], dim=1)).chunk(2, dim=1)
        shape = (feature.shape[0], self.channels) + (1,) * (feature.ndim - 2)
        gamma = torch.tanh(gamma).view(shape)
        beta = beta.view(shape)
        return feature * (1.0 + 0.1 * gamma) + 0.1 * beta


class DiseaseContextCrossAttention(nn.Module):
    """Use the disease latent as a query over context and image-summary tokens."""

    def __init__(self, latent_dim: int, feature_channels: Sequence[int], num_heads: int = 4) -> None:
        super().__init__()
        latent_dim = int(latent_dim)
        self.feature_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(int(channel)), nn.Linear(int(channel), latent_dim))
            for channel in feature_channels
        )
        self.context_token = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim))
        self.attention = nn.MultiheadAttention(latent_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, z_d: Tensor, z_c: Tensor, features: Sequence[Tensor]) -> Tensor:
        tokens = [self.context_token(z_c)]
        for feature, projection in zip(features, self.feature_projections, strict=True):
            pooled = feature.mean(dim=tuple(range(2, feature.ndim)))
            tokens.append(projection(pooled))
        key_value = torch.stack(tokens, dim=1)
        query = z_d.unsqueeze(1)
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        attended = self.norm(attended.squeeze(1) + z_d)
        return self.mlp(torch.cat([attended, z_c], dim=1))


def region_targets_from_subregions(mask: Tensor) -> Tensor:
    """Convert [NCR/NET, edema, ET] targets to [WT, TC, ET]."""

    ncr_net = mask[:, 0]
    edema = mask[:, 1]
    enhancing = mask[:, 2]
    whole_tumor = torch.amax(torch.stack([ncr_net, edema, enhancing], dim=1), dim=1)
    tumor_core = torch.amax(torch.stack([ncr_net, enhancing], dim=1), dim=1)
    return torch.stack([whole_tumor, tumor_core, enhancing], dim=1).clamp(0.0, 1.0)


def boundary_targets_from_subregions(mask: Tensor, kernel_size: int = 3) -> Tensor:
    """Build a binary tumor-boundary target from the union foreground mask."""

    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd.")
    foreground = mask.amax(dim=1, keepdim=True).float()
    padding = kernel_size // 2
    dilated = torch.nn.functional.max_pool3d(foreground, kernel_size, stride=1, padding=padding)
    eroded = 1.0 - torch.nn.functional.max_pool3d(1.0 - foreground, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)
