from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected a pair, got {value!r}")
    return int(value[0]), int(value[1])


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = ConvBlock(in_channels, channels[0])
        self.down = nn.ModuleList(
            [
                nn.Sequential(nn.MaxPool2d(2), ConvBlock(channels[0], channels[1])),
                nn.Sequential(nn.MaxPool2d(2), ConvBlock(channels[1], channels[2])),
                nn.Sequential(nn.MaxPool2d(2), ConvBlock(channels[2], channels[3])),
                nn.Sequential(nn.MaxPool2d(2), ConvBlock(channels[3], channels[3])),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[3], latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        for block in self.down:
            h = block(h)
        return self.proj(self.pool(h))


class SpatialDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        image_size: int | Sequence[int],
        base_channels: int = 32,
        output_activation: str | None = None,
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
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
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
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.head_uses_context = head_uses_context
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        self.disease_encoder = ImageEncoder(in_channels, latent_dim, base_channels)
        self.context_encoder = ImageEncoder(in_channels, latent_dim, base_channels)

        head_dim = latent_dim * 2 if head_uses_context else latent_dim
        self.classifier = MLPHead(head_dim, num_classes) if num_classes > 0 else None
        self.segmenter = (
            SpatialDecoder(head_dim, num_seg_classes, image_size, base_channels)
            if num_seg_classes > 0
            else None
        )
        self.reconstructor = SpatialDecoder(
            latent_dim * 2,
            in_channels,
            image_size,
            base_channels,
            output_activation="sigmoid",
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.disease_encoder(x), self.context_encoder(x)

    def _head_input(self, z_d: Tensor, z_c: Tensor) -> Tensor:
        if self.head_uses_context:
            return torch.cat([z_d, z_c], dim=1)
        return z_d

    def decode(self, z_d: Tensor, z_c: Tensor) -> Tensor:
        return self.reconstructor(torch.cat([z_d, z_c], dim=1))

    def predict_from_latents(self, z_d: Tensor, z_c: Tensor) -> dict[str, Tensor]:
        h = self._head_input(z_d, z_c)
        out: dict[str, Tensor] = {}
        if self.classifier is not None:
            out["logits"] = self.classifier(h)
        if self.segmenter is not None:
            out["seg_logits"] = self.segmenter(h)
        return out

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        z_d, z_c = self.encode(x)
        out = {
            "z_d": z_d,
            "z_c": z_c,
            "reconstruction": self.decode(z_d, z_c),
        }
        out.update(self.predict_from_latents(z_d, z_c))
        return out

