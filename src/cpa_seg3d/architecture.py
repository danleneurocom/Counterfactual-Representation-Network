from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from baselines.segformer3d import SegFormer3D
from cpa_seg3d.causal_blocks import DiseaseContextCrossAttention
from cpa_seg3d.decoder import CPARegionDecoder, LiteCPARegionDecoder


class CPASeg3D(nn.Module):
    """Causal Proxy-Adjusted 3D Segmentation Transformer.

    CPA-Seg3D keeps the SegFormer3D hierarchical transformer encoder as the
    representation backbone, then replaces the lightweight decoder with a
    causal, region-aware 3D decoder. The explicit intervention API estimates a
    model-level context-adjusted prediction by holding `z_d` fixed and averaging
    over a bank of training contexts `z_c`.
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
        latent_dim: int = 128,
        decoder_channels: int = 64,
        num_classes: int = 3,
        num_region_classes: int = 3,
        context_proxy_dim: int = 0,
        disease_proxy_dim: int = 0,
        annotation_proxy_dim: int = 0,
        deep_supervision: bool = True,
        decoder_variant: str = "lite",
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
        self.decoder_variant = str(decoder_variant)
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
            decoder_head_embedding_dim=max(8, int(decoder_channels)),
            num_classes=num_classes,
            decoder_dropout=0.0,
        )
        for parameter in self.backbone.segformer_decoder.parameters():
            parameter.requires_grad = False
        bottleneck_channels = int(embed_dims[-1])
        self.disease_head = self._latent_head(bottleneck_channels, self.latent_dim)
        self.context_head = self._latent_head(bottleneck_channels, self.latent_dim)
        self.cross_attention = DiseaseContextCrossAttention(self.latent_dim, self.feature_channels)
        decoder_cls: type[CPARegionDecoder | LiteCPARegionDecoder]
        if self.decoder_variant == "lite":
            decoder_cls = LiteCPARegionDecoder
        elif self.decoder_variant == "unet":
            decoder_cls = CPARegionDecoder
        else:
            raise ValueError(f"Unknown CPA-Seg3D decoder_variant: {self.decoder_variant}")
        self.decoder = decoder_cls(
            encoder_channels=self.feature_channels,
            latent_dim=self.latent_dim,
            decoder_channels=decoder_channels,
            num_subregion_classes=num_classes,
            num_region_classes=num_region_classes,
            deep_supervision=deep_supervision,
        )
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

    def predict_proxies(self, z_d: Tensor, z_c: Tensor) -> dict[str, Tensor]:
        predictions: dict[str, Tensor] = {}
        if self.context_proxy_head is not None:
            predictions["context_proxy_logits"] = self.context_proxy_head(z_c)
        if self.disease_proxy_head is not None:
            predictions["disease_proxy_logits"] = self.disease_proxy_head(z_d)
        if self.annotation_proxy_head is not None:
            predictions["annotation_proxy_logits"] = self.annotation_proxy_head(z_c)
        return predictions

    def decode_from_latents(
        self,
        features: Sequence[Tensor],
        z_d: Tensor,
        z_c: Tensor,
        output_size: tuple[int, int, int],
    ) -> dict[str, Tensor | list[Tensor]]:
        z_dc = self.cross_attention(z_d, z_c, features)
        outputs = self.decoder(features, z_d, z_c, z_dc, output_size)
        outputs["z_dc"] = z_dc
        return outputs

    def segment_from_latents(
        self,
        features: Sequence[Tensor],
        z_d: Tensor,
        z_c: Tensor,
        output_size: tuple[int, int, int] | None = None,
    ) -> Tensor:
        if output_size is None:
            c1_size = features[0].shape[2:]
            output_size = tuple(int(size) * 4 for size in c1_size)
        return self.decode_from_latents(features, z_d, z_c, output_size)["logits"]  # type: ignore[index]

    def backdoor_adjusted_outputs(
        self,
        features: Sequence[Tensor],
        z_d: Tensor,
        context_bank: Tensor,
        output_size: tuple[int, int, int],
        max_contexts: int | None = None,
    ) -> dict[str, Tensor]:
        if context_bank.ndim != 2:
            raise ValueError(f"context_bank must have shape [K, latent_dim], got {tuple(context_bank.shape)}")
        if context_bank.shape[1] != z_d.shape[1]:
            raise ValueError(f"context_bank latent dim {context_bank.shape[1]} does not match z_d dim {z_d.shape[1]}")
        bank = context_bank.to(device=z_d.device, dtype=z_d.dtype)
        if max_contexts is not None and max_contexts > 0 and bank.shape[0] > max_contexts:
            positions = torch.linspace(0, bank.shape[0] - 1, steps=max_contexts, device=bank.device)
            bank = bank[positions.round().long()]

        logits: list[Tensor] = []
        region_logits: list[Tensor] = []
        for context in bank:
            z_c = context.unsqueeze(0).expand(z_d.shape[0], -1)
            decoded = self.decode_from_latents(features, z_d, z_c, output_size)
            logits.append(decoded["logits"])  # type: ignore[arg-type]
            region_logits.append(decoded["region_logits"])  # type: ignore[arg-type]
        return {
            "adjusted_logits": torch.stack(logits, dim=0).mean(dim=0),
            "adjusted_region_logits": torch.stack(region_logits, dim=0).mean(dim=0),
        }

    def forward(
        self,
        x: Tensor,
        context_bank: Tensor | None = None,
        max_adjustment_contexts: int | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...] | list[Tensor]]:
        output_size = tuple(int(size) for size in x.shape[2:])
        features = self.encode_features(x)
        z_d, z_c = self.encode_latents(features)
        outputs = self.decode_from_latents(features, z_d, z_c, output_size)
        outputs.update(
            {
                "features": features,
                "z_d": z_d,
                "z_c": z_c,
            }
        )
        outputs.update(self.predict_proxies(z_d, z_c))
        if context_bank is not None:
            outputs.update(
                self.backdoor_adjusted_outputs(
                    features,
                    z_d,
                    context_bank,
                    output_size,
                    max_contexts=max_adjustment_contexts,
                )
            )
        return outputs

    def load_segformer3d_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Any]:
        """Load compatible SegFormer3D weights and report skipped keys."""

        current = self.backbone.state_dict()
        compatible: dict[str, Tensor] = {}
        skipped: list[str] = []
        for key, value in state_dict.items():
            stripped = key.removeprefix("backbone.")
            if stripped in current and tuple(current[stripped].shape) == tuple(value.shape):
                compatible[stripped] = value
            else:
                skipped.append(key)
        missing, unexpected = self.backbone.load_state_dict(compatible, strict=False)
        return {
            "loaded": len(compatible),
            "skipped": len(skipped),
            "missing": list(missing),
            "unexpected": list(unexpected),
        }


def build_cpa_seg3d_tiny(
    latent_dim: int = 16,
    num_classes: int = 3,
    context_proxy_dim: int = 0,
    disease_proxy_dim: int = 0,
    annotation_proxy_dim: int = 0,
    decoder_variant: str = "lite",
) -> CPASeg3D:
    return CPASeg3D(
        in_channels=4,
        sr_ratios=[4, 2, 1, 1],
        embed_dims=[4, 8, 16, 32],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[1, 1, 2, 4],
        depths=[1, 1, 1, 1],
        latent_dim=latent_dim,
        decoder_channels=16,
        num_classes=num_classes,
        context_proxy_dim=context_proxy_dim,
        disease_proxy_dim=disease_proxy_dim,
        annotation_proxy_dim=annotation_proxy_dim,
        decoder_variant=decoder_variant,
    )


def build_cpa_seg3d_base(
    latent_dim: int = 128,
    num_classes: int = 3,
    context_proxy_dim: int = 0,
    disease_proxy_dim: int = 0,
    annotation_proxy_dim: int = 0,
    decoder_variant: str = "lite",
) -> CPASeg3D:
    return CPASeg3D(
        in_channels=4,
        latent_dim=latent_dim,
        decoder_channels=64,
        num_classes=num_classes,
        context_proxy_dim=context_proxy_dim,
        disease_proxy_dim=disease_proxy_dim,
        annotation_proxy_dim=annotation_proxy_dim,
        decoder_variant=decoder_variant,
    )


def build_cpa_seg3d(
    model_size: str,
    latent_dim: int = 128,
    num_classes: int = 3,
    context_proxy_dim: int = 0,
    disease_proxy_dim: int = 0,
    annotation_proxy_dim: int = 0,
    decoder_variant: str = "lite",
) -> CPASeg3D:
    if model_size == "tiny":
        return build_cpa_seg3d_tiny(
            latent_dim=latent_dim,
            num_classes=num_classes,
            context_proxy_dim=context_proxy_dim,
            disease_proxy_dim=disease_proxy_dim,
            annotation_proxy_dim=annotation_proxy_dim,
            decoder_variant=decoder_variant,
        )
    if model_size == "base":
        return build_cpa_seg3d_base(
            latent_dim=latent_dim,
            num_classes=num_classes,
            context_proxy_dim=context_proxy_dim,
            disease_proxy_dim=disease_proxy_dim,
            annotation_proxy_dim=annotation_proxy_dim,
            decoder_variant=decoder_variant,
        )
    raise ValueError(f"Unknown CPA-Seg3D model_size: {model_size}")
