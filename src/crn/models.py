from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected a pair, got {value!r}")
    return int(value[0]), int(value[1])


def _resolve_group_count(num_channels: int, requested_groups: int) -> int:
    groups = max(1, min(int(requested_groups), int(num_channels)))
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


def _make_norm(num_channels: int, norm_type: str, group_norm_groups: int) -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        return nn.GroupNorm(_resolve_group_count(num_channels, group_norm_groups), num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm_type, group_norm_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm_type, group_norm_groups),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        base_channels: int = 32,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = ConvBlock(in_channels, channels[0], norm_type=norm_type, group_norm_groups=group_norm_groups)
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.MaxPool2d(2),
                    ConvBlock(channels[0], channels[1], norm_type=norm_type, group_norm_groups=group_norm_groups),
                ),
                nn.Sequential(
                    nn.MaxPool2d(2),
                    ConvBlock(channels[1], channels[2], norm_type=norm_type, group_norm_groups=group_norm_groups),
                ),
                nn.Sequential(
                    nn.MaxPool2d(2),
                    ConvBlock(channels[2], channels[3], norm_type=norm_type, group_norm_groups=group_norm_groups),
                ),
                nn.Sequential(
                    nn.MaxPool2d(2),
                    ConvBlock(channels[3], channels[3], norm_type=norm_type, group_norm_groups=group_norm_groups),
                ),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[3], latent_dim),
            nn.LayerNorm(latent_dim),
        )

    @property
    def feature_channels(self) -> list[int]:
        channels = [self.stem.net[0].out_channels]
        channels.extend(block[1].net[0].out_channels for block in self.down)
        return channels

    def forward_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        features: list[Tensor] = []
        h = self.stem(x)
        features.append(h)
        for block in self.down:
            h = block(h)
            features.append(h)
        return self.proj(self.pool(h)), features

    def forward(self, x: Tensor) -> Tensor:
        latent, _ = self.forward_features(x)
        return latent


class SpatialDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        image_size: int | Sequence[int],
        base_channels: int = 32,
        output_activation: str | None = None,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.image_size = _pair(image_size)
        self.output_activation = output_activation
        start_h = max(4, self.image_size[0] // 16)
        start_w = max(4, self.image_size[1] // 16)
        hidden = base_channels * 8
        self.start_shape = (hidden, start_h, start_w)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden * start_h * start_w),
            nn.SiLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden, base_channels * 4, kernel_size=4, stride=2, padding=1),
            _make_norm(base_channels * 4, norm_type, group_norm_groups),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            _make_norm(base_channels * 2, norm_type, group_norm_groups),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            _make_norm(base_channels, norm_type, group_norm_groups),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            _make_norm(base_channels, norm_type, group_norm_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(z.shape[0], *self.start_shape)
        out = self.up(h)
        if out.shape[-2:] != self.image_size:
            out = F.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(out)
        if self.output_activation == "tanh":
            return torch.tanh(out)
        return out


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, 128)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ContextualUNetDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_channels: Sequence[int],
        out_channels: int,
        image_size: int | Sequence[int],
        head_uses_context: bool = True,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(feature_channels) < 5:
            raise ValueError(f"Expected five encoder feature resolutions, got {feature_channels!r}")
        self.image_size = _pair(image_size)
        bottleneck_channels = int(feature_channels[-1])
        cond_dim = latent_dim * 2 if head_uses_context else latent_dim
        self.condition = nn.Sequential(
            nn.Linear(cond_dim, bottleneck_channels),
            nn.SiLU(inplace=True),
            nn.Linear(bottleneck_channels, bottleneck_channels),
        )
        self.blocks = nn.ModuleList(
            [
                UpBlock(
                    int(feature_channels[-1]),
                    int(feature_channels[-2]),
                    int(feature_channels[-2]),
                    norm_type=norm_type,
                    group_norm_groups=group_norm_groups,
                ),
                UpBlock(
                    int(feature_channels[-2]),
                    int(feature_channels[-3]),
                    int(feature_channels[-3]),
                    norm_type=norm_type,
                    group_norm_groups=group_norm_groups,
                ),
                UpBlock(
                    int(feature_channels[-3]),
                    int(feature_channels[-4]),
                    int(feature_channels[-4]),
                    norm_type=norm_type,
                    group_norm_groups=group_norm_groups,
                ),
                UpBlock(
                    int(feature_channels[-4]),
                    int(feature_channels[-5]),
                    int(feature_channels[-5]),
                    norm_type=norm_type,
                    group_norm_groups=group_norm_groups,
                ),
            ]
        )
        self.out = nn.Conv2d(int(feature_channels[0]), out_channels, kernel_size=1)

    def forward_with_features(self, features: Sequence[Tensor], head_latent: Tensor) -> tuple[Tensor, Tensor]:
        if len(features) < 5:
            raise ValueError(f"Expected five encoder feature tensors, got {len(features)}")
        h = features[-1] + self.condition(head_latent).unsqueeze(-1).unsqueeze(-1)
        skip_features = list(reversed(features[:-1]))
        for block, skip in zip(self.blocks, skip_features, strict=True):
            h = block(h, skip)
        logits = self.out(h)
        if logits.shape[-2:] != self.image_size:
            logits = F.interpolate(logits, size=self.image_size, mode="bilinear", align_corners=False)
        return logits, h

    def forward(self, features: Sequence[Tensor], head_latent: Tensor) -> Tensor:
        logits, _ = self.forward_with_features(features, head_latent)
        return logits


class CounterfactualRepresentationNetwork(nn.Module):
    """Dual-latent CRN with corrected confounder-aware prediction heads."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        image_size: int | Sequence[int] = (128, 128),
        latent_dim: int = 128,
        base_channels: int = 32,
        num_seg_classes: int = 0,
        head_uses_context: bool = True,
        segmentation_head: str = "latent",
        init_volume_context_weight: float = 0.5,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.head_uses_context = head_uses_context
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.segmentation_head = segmentation_head

        self.disease_encoder = ImageEncoder(
            in_channels,
            latent_dim,
            base_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.context_encoder = ImageEncoder(
            in_channels,
            latent_dim,
            base_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        init_volume_context_weight = float(min(max(init_volume_context_weight, 1e-4), 1.0 - 1e-4))
        self.volume_context_logit = nn.Parameter(
            torch.tensor(math.log(init_volume_context_weight / (1.0 - init_volume_context_weight)), dtype=torch.float32)
        )

        head_dim = latent_dim * 2 if head_uses_context else latent_dim
        self.classifier = MLPHead(head_dim, num_classes) if num_classes > 0 else None
        if num_seg_classes > 0:
            if segmentation_head == "latent":
                self.segmenter = SpatialDecoder(
                    head_dim,
                    num_seg_classes,
                    image_size,
                    base_channels,
                    norm_type=norm_type,
                    group_norm_groups=group_norm_groups,
                )
            elif segmentation_head == "unet":
                self.segmenter = ContextualUNetDecoder(
                    latent_dim=latent_dim,
                    feature_channels=self.disease_encoder.feature_channels,
                    out_channels=num_seg_classes,
                    image_size=image_size,
                    head_uses_context=head_uses_context,
                    norm_type=norm_type,
                    group_norm_groups=group_norm_groups,
                )
            else:
                raise ValueError(f"Unknown segmentation_head: {segmentation_head}")
        else:
            self.segmenter = None
        self.reconstructor = SpatialDecoder(
            latent_dim * 2,
            in_channels,
            image_size,
            base_channels,
            output_activation="sigmoid",
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.supports_latent_segmentation = segmentation_head == "latent"

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.disease_encoder(x), self.context_encoder(x)

    def _head_input(self, z_d: Tensor, z_c: Tensor) -> Tensor:
        if self.head_uses_context:
            return torch.cat([z_d, z_c], dim=1)
        return z_d

    def fuse_context_latents(self, z_c_slice: Tensor, z_c_volume: Tensor | None) -> Tensor:
        if z_c_volume is None:
            return z_c_slice
        weight = torch.sigmoid(self.volume_context_logit)
        fused = (1.0 - weight) * z_c_slice + weight * z_c_volume
        return F.layer_norm(fused, (fused.shape[-1],))

    def decode(self, z_d: Tensor, z_c: Tensor) -> Tensor:
        return self.reconstructor(torch.cat([z_d, z_c], dim=1))

    def segment_from_parts(
        self,
        z_d: Tensor,
        z_c: Tensor,
        disease_features: Sequence[Tensor] | None = None,
    ) -> Tensor:
        return self.segment_outputs_from_parts(z_d, z_c, disease_features)["seg_logits"]

    def segment_outputs_from_parts(
        self,
        z_d: Tensor,
        z_c: Tensor,
        disease_features: Sequence[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if self.segmenter is None:
            raise ValueError("Segmentation head is not configured for this model.")
        h = self._head_input(z_d, z_c)
        if self.supports_latent_segmentation:
            seg_logits = self.segmenter(h)
            return {"seg_logits": seg_logits, "seg_features": seg_logits}
        if disease_features is None:
            raise ValueError("U-Net segmentation head requires disease feature maps.")
        seg_logits, seg_features = self.segmenter.forward_with_features(disease_features, h)
        return {"seg_logits": seg_logits, "seg_features": seg_features}

    def predict_from_latents(self, z_d: Tensor, z_c: Tensor) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        if self.classifier is not None:
            out["logits"] = self.classifier(self._head_input(z_d, z_c))
        return out

    def refresh_outputs(
        self,
        outputs: dict[str, Tensor],
        z_c: Tensor,
        volume_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        refreshed = dict(outputs)
        refreshed["z_c"] = z_c
        refreshed["reconstruction"] = self.decode(outputs["z_d"], z_c)
        if volume_context is not None:
            refreshed["volume_context"] = volume_context
        refreshed.update(self.predict_from_latents(outputs["z_d"], z_c))
        if self.segmenter is not None:
            refreshed.update(self.segment_outputs_from_parts(outputs["z_d"], z_c, outputs.get("disease_features")))
        return refreshed

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        z_d, disease_features = self.disease_encoder.forward_features(x)
        z_c_slice = self.context_encoder(x)
        out = {
            "z_d": z_d,
            "z_c_slice": z_c_slice,
            "z_c": z_c_slice,
            "disease_features": tuple(disease_features),
            "reconstruction": self.decode(z_d, z_c_slice),
        }
        out.update(self.predict_from_latents(z_d, z_c_slice))
        if self.segmenter is not None:
            out.update(self.segment_outputs_from_parts(z_d, z_c_slice, disease_features))
        return out
