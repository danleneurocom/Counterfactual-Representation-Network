from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from baselines.segformer3d.architectures.segformer3d import SegFormer3D


class LatentFeatureModulator(nn.Module):
    """FiLM-style feature modulation by disease/context latents.

    The final projection is zero-initialized so the causal wrapper initially
    behaves like the baseline decoder. This makes it possible to warm-start from
    a trained SegFormer3D checkpoint and then learn causal modulation.
    """

    def __init__(self, feature_channels: Sequence[int], latent_dim: int, modulation_scale: float = 0.1) -> None:
        super().__init__()
        self.feature_channels = tuple(int(channel) for channel in feature_channels)
        self.total_channels = sum(self.feature_channels)
        self.modulation_scale = float(modulation_scale)
        self.proj = nn.Linear(int(latent_dim) * 2, self.total_channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: Sequence[Tensor], z_d: Tensor, z_c: Tensor) -> tuple[Tensor, ...]:
        if len(features) != len(self.feature_channels):
            raise ValueError(f"Expected {len(self.feature_channels)} feature maps, got {len(features)}")
        if z_d.shape != z_c.shape:
            raise ValueError(f"z_d and z_c must have matching shapes, got {tuple(z_d.shape)} and {tuple(z_c.shape)}")

        params = self.proj(torch.cat([z_d, z_c], dim=1))
        gamma_all, beta_all = params.chunk(2, dim=1)
        gammas = gamma_all.split(self.feature_channels, dim=1)
        betas = beta_all.split(self.feature_channels, dim=1)

        modulated = []
        for feature, gamma, beta, channels in zip(features, gammas, betas, self.feature_channels, strict=True):
            if feature.shape[1] != channels:
                raise ValueError(f"Expected feature with {channels} channels, got {feature.shape[1]}")
            shape = (feature.shape[0], channels) + (1,) * (feature.ndim - 2)
            gamma = torch.tanh(gamma).view(shape)
            beta = beta.view(shape)
            modulated.append(feature * (1.0 + self.modulation_scale * gamma) + self.modulation_scale * beta)
        return tuple(modulated)


class CausalSegFormer3D(nn.Module):
    """SegFormer3D with explicit disease/context proxy latents.

    This is an estimator scaffold for Pearl-style queries, not by itself proof
    of real-world causal identification. The intervention hooks are meaningful
    only under the SCM and proxy assumptions documented in
    `docs/segformer3d_causal_phase.md`.
    """

    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list[int] | None = None,
        embed_dims: list[int] | None = None,
        patch_kernel_size: list[int] | None = None,
        patch_stride: list[int] | None = None,
        patch_padding: list[int] | None = None,
        mlp_ratios: list[int] | None = None,
        num_heads: list[int] | None = None,
        depths: list[int] | None = None,
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        decoder_dropout: float = 0.0,
        latent_dim: int = 128,
        context_proxy_dim: int = 0,
        disease_proxy_dim: int = 0,
        annotation_proxy_dim: int = 0,
    ) -> None:
        super().__init__()
        sr_ratios = sr_ratios or [4, 2, 1, 1]
        embed_dims = embed_dims or [32, 64, 160, 256]
        patch_kernel_size = patch_kernel_size or [7, 3, 3, 3]
        patch_stride = patch_stride or [4, 2, 2, 2]
        patch_padding = patch_padding or [3, 1, 1, 1]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        num_heads = num_heads or [1, 2, 5, 8]
        depths = depths or [2, 2, 2, 2]

        self.latent_dim = int(latent_dim)
        self.context_proxy_dim = int(context_proxy_dim)
        self.disease_proxy_dim = int(disease_proxy_dim)
        self.annotation_proxy_dim = int(annotation_proxy_dim)
        self.feature_channels = tuple(int(channel) for channel in embed_dims)
        self.backbone = SegFormer3D(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            decoder_dropout=decoder_dropout,
        )
        bottleneck_channels = int(embed_dims[-1])
        self.disease_head = self._latent_head(bottleneck_channels, self.latent_dim)
        self.context_head = self._latent_head(bottleneck_channels, self.latent_dim)
        self.modulator = LatentFeatureModulator(self.feature_channels, self.latent_dim)
        self.context_proxy_head = self._proxy_head(self.latent_dim, self.context_proxy_dim)
        self.disease_proxy_head = self._proxy_head(self.latent_dim, self.disease_proxy_dim)
        self.annotation_proxy_head = self._proxy_head(self.latent_dim, self.annotation_proxy_dim)

    @staticmethod
    def _latent_head(in_features: int, latent_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    @staticmethod
    def _proxy_head(latent_dim: int, out_features: int) -> nn.Module | None:
        if out_features <= 0:
            return None
        return nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, out_features),
        )

    def encode_features(self, x: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.backbone.segformer_encoder(x))

    def encode_latents(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        bottleneck = features[-1].mean(dim=(2, 3, 4))
        return self.disease_head(bottleneck), self.context_head(bottleneck)

    def segment_from_latents(self, features: Sequence[Tensor], z_d: Tensor, z_c: Tensor) -> Tensor:
        modulated = self.modulator(features, z_d, z_c)
        c1, c2, c3, c4 = modulated
        return self.backbone.segformer_decoder(c1, c2, c3, c4)

    def predict_proxies(self, z_d: Tensor, z_c: Tensor) -> dict[str, Tensor]:
        predictions: dict[str, Tensor] = {}
        if self.context_proxy_head is not None:
            predictions["context_proxy_logits"] = self.context_proxy_head(z_c)
        if self.disease_proxy_head is not None:
            predictions["disease_proxy_logits"] = self.disease_proxy_head(z_d)
        if self.annotation_proxy_head is not None:
            predictions["annotation_proxy_logits"] = self.annotation_proxy_head(z_c)
        return predictions

    def backdoor_adjusted_logits(
        self,
        features: Sequence[Tensor],
        z_d: Tensor,
        context_bank: Tensor,
        max_contexts: int | None = None,
    ) -> Tensor:
        if context_bank.ndim != 2:
            raise ValueError(f"context_bank must have shape [K, latent_dim], got {tuple(context_bank.shape)}")
        if context_bank.shape[1] != z_d.shape[1]:
            raise ValueError(f"context_bank latent dim {context_bank.shape[1]} does not match z_d dim {z_d.shape[1]}")
        bank = context_bank.to(device=z_d.device, dtype=z_d.dtype)
        if max_contexts is not None and max_contexts > 0 and bank.shape[0] > max_contexts:
            positions = torch.linspace(0, bank.shape[0] - 1, steps=max_contexts, device=bank.device)
            bank = bank[positions.round().long()]
        logits = []
        for context in bank:
            z_c = context.unsqueeze(0).expand(z_d.shape[0], -1)
            logits.append(self.segment_from_latents(features, z_d, z_c))
        return torch.stack(logits, dim=0).mean(dim=0)

    def forward(
        self,
        x: Tensor,
        context_bank: Tensor | None = None,
        max_adjustment_contexts: int | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        features = self.encode_features(x)
        z_d, z_c = self.encode_latents(features)
        logits = self.segment_from_latents(features, z_d, z_c)
        outputs: dict[str, Tensor | tuple[Tensor, ...]] = {
            "logits": logits,
            "z_d": z_d,
            "z_c": z_c,
            "features": features,
        }
        outputs.update(self.predict_proxies(z_d, z_c))
        if context_bank is not None:
            outputs["adjusted_logits"] = self.backdoor_adjusted_logits(
                features,
                z_d,
                context_bank,
                max_contexts=max_adjustment_contexts,
            )
        return outputs

    def load_baseline_state_dict(self, state_dict: dict[str, Tensor], strict_backbone: bool = True) -> None:
        """Load a trained `SegFormer3D` state dict into the shared backbone."""
        self.backbone.load_state_dict(state_dict, strict=strict_backbone)


def build_causal_segformer3d_tiny(
    latent_dim: int = 16,
    num_classes: int = 3,
    context_proxy_dim: int = 0,
    disease_proxy_dim: int = 0,
    annotation_proxy_dim: int = 0,
) -> CausalSegFormer3D:
    return CausalSegFormer3D(
        in_channels=4,
        sr_ratios=[4, 2, 1, 1],
        embed_dims=[4, 8, 16, 32],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[1, 1, 2, 4],
        depths=[1, 1, 1, 1],
        decoder_head_embedding_dim=8,
        num_classes=num_classes,
        decoder_dropout=0.0,
        latent_dim=latent_dim,
        context_proxy_dim=context_proxy_dim,
        disease_proxy_dim=disease_proxy_dim,
        annotation_proxy_dim=annotation_proxy_dim,
    )


def build_causal_segformer3d_base(
    latent_dim: int = 128,
    num_classes: int = 3,
    context_proxy_dim: int = 0,
    disease_proxy_dim: int = 0,
    annotation_proxy_dim: int = 0,
) -> CausalSegFormer3D:
    return CausalSegFormer3D(
        in_channels=4,
        num_classes=num_classes,
        latent_dim=latent_dim,
        context_proxy_dim=context_proxy_dim,
        disease_proxy_dim=disease_proxy_dim,
        annotation_proxy_dim=annotation_proxy_dim,
    )


def build_causal_segformer3d(
    model_size: str,
    latent_dim: int = 128,
    num_classes: int = 3,
    context_proxy_dim: int = 0,
    disease_proxy_dim: int = 0,
    annotation_proxy_dim: int = 0,
) -> CausalSegFormer3D:
    if model_size == "tiny":
        return build_causal_segformer3d_tiny(
            latent_dim=latent_dim,
            num_classes=num_classes,
            context_proxy_dim=context_proxy_dim,
            disease_proxy_dim=disease_proxy_dim,
            annotation_proxy_dim=annotation_proxy_dim,
        )
    if model_size == "base":
        return build_causal_segformer3d_base(
            latent_dim=latent_dim,
            num_classes=num_classes,
            context_proxy_dim=context_proxy_dim,
            disease_proxy_dim=disease_proxy_dim,
            annotation_proxy_dim=annotation_proxy_dim,
        )
    raise ValueError(f"Unknown causal SegFormer3D model_size: {model_size}")
