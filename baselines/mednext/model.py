from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from crn.mednext_blocks import MedNeXtBlock, MedNeXtDownBlock


MEDNEXT_SEGMENTER_CONFIGS: dict[str, dict[str, object]] = {
    "S": {
        "base_channels": 32,
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "exp_ratios": [2, 2, 2, 2, 2, 2, 2, 2, 2],
    },
    "B": {
        "base_channels": 32,
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "exp_ratios": [2, 3, 4, 4, 4, 4, 4, 3, 2],
    },
    "M": {
        "base_channels": 32,
        "block_counts": [3, 4, 4, 4, 4, 4, 4, 4, 3],
        "exp_ratios": [2, 3, 4, 4, 4, 4, 4, 3, 2],
    },
    "L": {
        "base_channels": 32,
        "block_counts": [3, 4, 8, 8, 8, 8, 8, 4, 3],
        "exp_ratios": [3, 4, 8, 8, 8, 8, 8, 4, 3],
    },
}


class MedNeXtUpBlock(nn.Module):
    """Residual inverted-bottleneck 2x upsampling block for MedNeXt decoders."""

    def __init__(self, in_channels: int, out_channels: int, exp_ratio: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        hidden = in_channels * exp_ratio
        self.dwconv = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            groups=in_channels,
        )
        self.norm = nn.GroupNorm(num_groups=min(in_channels, 32), num_channels=in_channels)
        self.expand = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(hidden, out_channels, kernel_size=1)
        self.skip = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=2)

    @staticmethod
    def _match_spatial(x: Tensor, spatial_shape: tuple[int, int, int]) -> Tensor:
        if tuple(x.shape[-3:]) == spatial_shape:
            return x
        return F.interpolate(x, size=spatial_shape, mode="trilinear", align_corners=False)

    def forward(self, x: Tensor, output_size: tuple[int, int, int] | None = None) -> Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.expand(h)
        h = self.act(h)
        h = self.compress(h)
        skip = self.skip(x)
        if output_size is not None:
            h = self._match_spatial(h, output_size)
            skip = self._match_spatial(skip, output_size)
        return h + skip


def _stage(channels: int, exp_ratio: int, depth: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        *[
            MedNeXtBlock(
                in_channels=channels,
                out_channels=channels,
                exp_ratio=exp_ratio,
                kernel_size=kernel_size,
                dim=3,
            )
            for _ in range(depth)
        ]
    )


class MedNeXtSegmenter(nn.Module):
    """Full 3D MedNeXt encoder-decoder for multilabel BraTS-style segmentation."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        model_id: str = "S",
        kernel_size: int = 3,
        deep_supervision: bool = False,
        base_channels: int | None = None,
    ) -> None:
        super().__init__()
        key = model_id.upper()
        if key not in MEDNEXT_SEGMENTER_CONFIGS:
            raise ValueError(f"Unknown MedNeXt model_id {model_id!r}; choose from S, B, M, L.")
        config = MEDNEXT_SEGMENTER_CONFIGS[key]
        block_counts = [int(value) for value in config["block_counts"]]
        exp_ratios = [int(value) for value in config["exp_ratios"]]
        base = int(base_channels or config["base_channels"])
        channels = [base, base * 2, base * 4, base * 8, base * 16]

        self.model_id = key
        self.kernel_size = int(kernel_size)
        self.deep_supervision = bool(deep_supervision)
        self._feature_channels = tuple(channels)
        self.stem = nn.Conv3d(in_channels, channels[0], kernel_size=1)

        self.enc0 = _stage(channels[0], exp_ratios[0], block_counts[0], kernel_size)
        self.down0 = MedNeXtDownBlock(channels[0], channels[1], exp_ratios[1], kernel_size=kernel_size, dim=3)
        self.enc1 = _stage(channels[1], exp_ratios[1], block_counts[1], kernel_size)
        self.down1 = MedNeXtDownBlock(channels[1], channels[2], exp_ratios[2], kernel_size=kernel_size, dim=3)
        self.enc2 = _stage(channels[2], exp_ratios[2], block_counts[2], kernel_size)
        self.down2 = MedNeXtDownBlock(channels[2], channels[3], exp_ratios[3], kernel_size=kernel_size, dim=3)
        self.enc3 = _stage(channels[3], exp_ratios[3], block_counts[3], kernel_size)
        self.down3 = MedNeXtDownBlock(channels[3], channels[4], exp_ratios[4], kernel_size=kernel_size, dim=3)

        self.bottleneck = _stage(channels[4], exp_ratios[4], block_counts[4], kernel_size)

        self.up3 = MedNeXtUpBlock(channels[4], channels[3], exp_ratios[5], kernel_size=kernel_size)
        self.dec3 = _stage(channels[3], exp_ratios[5], block_counts[5], kernel_size)
        self.up2 = MedNeXtUpBlock(channels[3], channels[2], exp_ratios[6], kernel_size=kernel_size)
        self.dec2 = _stage(channels[2], exp_ratios[6], block_counts[6], kernel_size)
        self.up1 = MedNeXtUpBlock(channels[2], channels[1], exp_ratios[7], kernel_size=kernel_size)
        self.dec1 = _stage(channels[1], exp_ratios[7], block_counts[7], kernel_size)
        self.up0 = MedNeXtUpBlock(channels[1], channels[0], exp_ratios[8], kernel_size=kernel_size)
        self.dec0 = _stage(channels[0], exp_ratios[8], block_counts[8], kernel_size)

        self.out0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        if self.deep_supervision:
            self.out1 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
            self.out2 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
            self.out3 = nn.Conv3d(channels[3], num_classes, kernel_size=1)
            self.out4 = nn.Conv3d(channels[4], num_classes, kernel_size=1)

    @property
    def feature_channels(self) -> tuple[int, ...]:
        return self._feature_channels

    def encode_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x0 = self.enc0(self.stem(x))
        x1 = self.enc1(self.down0(x0))
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        xb = self.bottleneck(self.down3(x3))
        return x0, x1, x2, x3, xb

    def decode_feature_maps(
        self,
        features: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x0, x1, x2, x3, xb = features
        h3 = self.dec3(self.up3(xb, tuple(x3.shape[-3:])) + x3)
        h2 = self.dec2(self.up2(h3, tuple(x2.shape[-3:])) + x2)
        h1 = self.dec1(self.up1(h2, tuple(x1.shape[-3:])) + x1)
        h0 = self.dec0(self.up0(h1, tuple(x0.shape[-3:])) + x0)
        return h0, h1, h2, h3, xb

    def logits_from_decoder_features(
        self,
        decoder_features: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        output_shape: tuple[int, int, int],
    ) -> Tensor | list[Tensor]:
        h0, h1, h2, h3, xb = decoder_features
        logits = self.out0(h0)
        if tuple(logits.shape[-3:]) != tuple(output_shape):
            logits = F.interpolate(logits, size=output_shape, mode="trilinear", align_corners=False)
        if not self.deep_supervision:
            return logits
        return [logits, self.out1(h1), self.out2(h2), self.out3(h3), self.out4(xb)]

    def decode_features(self, features: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], output_shape: tuple[int, int, int]) -> Tensor | list[Tensor]:
        return self.logits_from_decoder_features(self.decode_feature_maps(features), output_shape)

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        return self.decode_features(self.encode_features(x), tuple(x.shape[-3:]))


def build_mednext_segmenter(
    model_id: str = "S",
    kernel_size: int = 3,
    in_channels: int = 4,
    num_classes: int = 3,
    deep_supervision: bool = False,
    base_channels: int | None = None,
) -> MedNeXtSegmenter:
    return MedNeXtSegmenter(
        in_channels=in_channels,
        num_classes=num_classes,
        model_id=model_id,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
        base_channels=base_channels,
    )
