from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cpa_seg3d.causal_blocks import CausalFiLM3D


def _groups(channels: int) -> int:
    for group_count in (8, 4, 2, 1):
        if channels % group_count == 0:
            return group_count
    return 1


class ConvRefineBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CPARegionDecoder(nn.Module):
    """U-Net-style region-aware decoder for CPA-Seg3D."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        latent_dim: int,
        decoder_channels: int = 64,
        num_subregion_classes: int = 3,
        num_region_classes: int = 3,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("CPARegionDecoder expects four encoder feature maps.")
        c1, c2, c3, c4 = [int(channel) for channel in encoder_channels]
        d = int(decoder_channels)
        self.deep_supervision = bool(deep_supervision)
        self.proj4 = nn.Conv3d(c4, d, kernel_size=1)
        self.proj3 = nn.Conv3d(c3, d, kernel_size=1)
        self.proj2 = nn.Conv3d(c2, d, kernel_size=1)
        self.proj1 = nn.Conv3d(c1, d, kernel_size=1)

        self.refine3 = ConvRefineBlock3D(d * 2, d)
        self.refine2 = ConvRefineBlock3D(d * 2, d)
        self.refine1 = ConvRefineBlock3D(d * 2, d)
        self.full_refine = ConvRefineBlock3D(d, d)

        self.mod3 = CausalFiLM3D(latent_dim, d)
        self.mod2 = CausalFiLM3D(latent_dim, d)
        self.mod1 = CausalFiLM3D(latent_dim, d)
        self.mod_full = CausalFiLM3D(latent_dim, d)

        self.subregion_head = nn.Conv3d(d, num_subregion_classes, kernel_size=1)
        self.region_head = nn.Conv3d(d, num_region_classes, kernel_size=1)
        self.boundary_head = nn.Conv3d(d, 1, kernel_size=1)

        if self.deep_supervision:
            self.deep_heads = nn.ModuleList(
                [
                    nn.Conv3d(d, num_subregion_classes, kernel_size=1),
                    nn.Conv3d(d, num_subregion_classes, kernel_size=1),
                    nn.Conv3d(d, num_subregion_classes, kernel_size=1),
                ]
            )
        else:
            self.deep_heads = nn.ModuleList()

    @staticmethod
    def _upsample_to(x: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(x, size=reference.shape[2:], mode="trilinear", align_corners=False)

    def forward(
        self,
        features: Sequence[Tensor],
        z_d: Tensor,
        z_c: Tensor,
        z_dc: Tensor,
        output_size: tuple[int, int, int],
    ) -> dict[str, Tensor | list[Tensor]]:
        c1, c2, c3, c4 = features
        x = self.proj4(c4)

        x = self._upsample_to(x, c3)
        x = self.refine3(torch.cat([x, self.proj3(c3)], dim=1))
        x = self.mod3(x, z_d, z_c, z_dc)
        deep3 = x

        x = self._upsample_to(x, c2)
        x = self.refine2(torch.cat([x, self.proj2(c2)], dim=1))
        x = self.mod2(x, z_d, z_c, z_dc)
        deep2 = x

        x = self._upsample_to(x, c1)
        x = self.refine1(torch.cat([x, self.proj1(c1)], dim=1))
        x = self.mod1(x, z_d, z_c, z_dc)
        deep1 = x

        x = F.interpolate(x, size=output_size, mode="trilinear", align_corners=False)
        x = self.full_refine(x)
        x = self.mod_full(x, z_d, z_c, z_dc)

        outputs: dict[str, Tensor | list[Tensor]] = {
            "decoder_features": x,
            "logits": self.subregion_head(x),
            "region_logits": self.region_head(x),
            "boundary_logits": self.boundary_head(x),
        }
        if self.deep_supervision:
            outputs["deep_logits"] = [
                F.interpolate(head(feature), size=output_size, mode="trilinear", align_corners=False)
                for head, feature in zip(self.deep_heads, (deep3, deep2, deep1), strict=True)
            ]
        return outputs


class LiteCPARegionDecoder(nn.Module):
    """Fast region-aware decoder that avoids full-resolution 3D refinement.

    All convolutional fusion happens at the first encoder stage resolution
    (`input / 4` for the current SegFormer3D encoder). The final subregion,
    region, boundary, and deep-supervision logits are then upsampled to the
    requested output size. This keeps the causal/region heads while removing the
    expensive full-resolution U-Net-style refinement block.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        latent_dim: int,
        decoder_channels: int = 64,
        num_subregion_classes: int = 3,
        num_region_classes: int = 3,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("LiteCPARegionDecoder expects four encoder feature maps.")
        c1, c2, c3, c4 = [int(channel) for channel in encoder_channels]
        d = int(decoder_channels)
        self.deep_supervision = bool(deep_supervision)
        self.proj1 = nn.Conv3d(c1, d, kernel_size=1)
        self.proj2 = nn.Conv3d(c2, d, kernel_size=1)
        self.proj3 = nn.Conv3d(c3, d, kernel_size=1)
        self.proj4 = nn.Conv3d(c4, d, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(d * 4, d, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(d), d),
            nn.SiLU(inplace=True),
            nn.Conv3d(d, d, kernel_size=3, padding=1, groups=max(1, _groups(d)), bias=False),
            nn.GroupNorm(_groups(d), d),
            nn.SiLU(inplace=True),
        )
        self.mod = CausalFiLM3D(latent_dim, d)
        self.subregion_head = nn.Conv3d(d, num_subregion_classes, kernel_size=1)
        self.region_head = nn.Conv3d(d, num_region_classes, kernel_size=1)
        self.boundary_head = nn.Conv3d(d, 1, kernel_size=1)
        if self.deep_supervision:
            self.deep_heads = nn.ModuleList(
                [
                    nn.Conv3d(d, num_subregion_classes, kernel_size=1),
                    nn.Conv3d(d, num_subregion_classes, kernel_size=1),
                    nn.Conv3d(d, num_subregion_classes, kernel_size=1),
                ]
            )
        else:
            self.deep_heads = nn.ModuleList()

    @staticmethod
    def _upsample_to(x: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(x, size=reference.shape[2:], mode="trilinear", align_corners=False)

    @staticmethod
    def _upsample_logits(logits: Tensor, output_size: tuple[int, int, int]) -> Tensor:
        return F.interpolate(logits, size=output_size, mode="trilinear", align_corners=False)

    def forward(
        self,
        features: Sequence[Tensor],
        z_d: Tensor,
        z_c: Tensor,
        z_dc: Tensor,
        output_size: tuple[int, int, int],
    ) -> dict[str, Tensor | list[Tensor]]:
        c1, c2, c3, c4 = features
        p1 = self.proj1(c1)
        p2 = self._upsample_to(self.proj2(c2), c1)
        p3 = self._upsample_to(self.proj3(c3), c1)
        p4 = self._upsample_to(self.proj4(c4), c1)
        x = self.fuse(torch.cat([p1, p2, p3, p4], dim=1))
        x = self.mod(x, z_d, z_c, z_dc)

        outputs: dict[str, Tensor | list[Tensor]] = {
            "decoder_features": x,
            "logits": self._upsample_logits(self.subregion_head(x), output_size),
            "region_logits": self._upsample_logits(self.region_head(x), output_size),
            "boundary_logits": self._upsample_logits(self.boundary_head(x), output_size),
        }
        if self.deep_supervision:
            outputs["deep_logits"] = [
                self._upsample_logits(head(feature), output_size)
                for head, feature in zip(self.deep_heads, (p4, p3, p2), strict=True)
            ]
        return outputs
