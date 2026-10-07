from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


BBox3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def subregion_to_region_prob(subregion_prob: Tensor) -> Tensor:
    ncr_net = subregion_prob[:, 0:1]
    edema = subregion_prob[:, 1:2]
    enhancing = subregion_prob[:, 2:3]
    whole_tumor = 1.0 - (1.0 - ncr_net) * (1.0 - edema) * (1.0 - enhancing)
    tumor_core = 1.0 - (1.0 - ncr_net) * (1.0 - enhancing)
    return torch.cat([whole_tumor, tumor_core, enhancing], dim=1).clamp(0.0, 1.0)


def bbox_from_mask(mask: Tensor, margin: int = 0, fallback_shape: Sequence[int] | None = None) -> BBox3D:
    if mask.ndim == 4:
        foreground = mask.detach().amax(dim=0) > 0.5
    elif mask.ndim == 3:
        foreground = mask.detach() > 0.5
    else:
        raise ValueError(f"Expected mask shape [C,D,H,W] or [D,H,W], got {tuple(mask.shape)}")
    shape = tuple(int(value) for value in foreground.shape)
    if not bool(foreground.any().detach().cpu()):
        fallback = tuple(int(value) for value in (fallback_shape or shape))
        return tuple((0, value) for value in fallback)  # type: ignore[return-value]
    coords = torch.nonzero(foreground, as_tuple=False)
    starts = coords.amin(dim=0) - int(margin)
    stops = coords.amax(dim=0) + int(margin) + 1
    starts = torch.maximum(starts, torch.zeros_like(starts))
    stops = torch.minimum(stops, torch.as_tensor(shape, device=stops.device, dtype=stops.dtype))
    return tuple((int(start), int(stop)) for start, stop in zip(starts.tolist(), stops.tolist(), strict=True))  # type: ignore[return-value]


def bbox_from_probabilities(probabilities: Tensor, threshold: float = 0.5, margin: int = 0) -> BBox3D:
    if probabilities.ndim != 4:
        raise ValueError(f"Expected probabilities shape [C,D,H,W], got {tuple(probabilities.shape)}")
    foreground = probabilities.detach().amax(dim=0) > float(threshold)
    if bool(foreground.any().detach().cpu()):
        return bbox_from_mask(foreground, margin=margin)
    score = probabilities.detach().amax(dim=0)
    flat_index = int(score.reshape(-1).argmax().item())
    z = flat_index // (score.shape[1] * score.shape[2])
    y = (flat_index // score.shape[2]) % score.shape[1]
    x = flat_index % score.shape[2]
    radius = max(2, int(margin))
    shape = tuple(int(value) for value in score.shape)
    return (
        (max(0, z - radius), min(shape[0], z + radius + 1)),
        (max(0, y - radius), min(shape[1], y + radius + 1)),
        (max(0, x - radius), min(shape[2], x + radius + 1)),
    )


def scale_bbox(bbox: BBox3D, source_shape: Sequence[int], target_shape: Sequence[int]) -> BBox3D:
    scaled = []
    for (start, stop), source, target in zip(bbox, source_shape, target_shape, strict=True):
        source = max(int(source), 1)
        target = max(int(target), 1)
        new_start = int(torch.floor(torch.tensor(start * target / source)).item())
        new_stop = int(torch.ceil(torch.tensor(stop * target / source)).item())
        new_start = max(0, min(new_start, target - 1))
        new_stop = max(new_start + 1, min(new_stop, target))
        scaled.append((new_start, new_stop))
    return tuple(scaled)  # type: ignore[return-value]


def crop_resize_3d(tensor: Tensor, bbox: BBox3D, size: int, mode: str) -> Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor shape [C,D,H,W], got {tuple(tensor.shape)}")
    slices = tuple(slice(start, stop) for start, stop in bbox)
    crop = tensor[(slice(None), *slices)].unsqueeze(0)
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    resized = F.interpolate(crop, size=(int(size), int(size), int(size)), mode=mode, **kwargs)
    return resized.squeeze(0)


def paste_resized_3d(base: Tensor, roi_logits: Tensor, bbox: BBox3D) -> Tensor:
    if base.ndim != 4 or roi_logits.ndim != 4:
        raise ValueError("base and roi_logits must both have shape [C,D,H,W].")
    slices = tuple(slice(start, stop) for start, stop in bbox)
    target_size = tuple(stop - start for start, stop in bbox)
    resized = F.interpolate(
        roi_logits.unsqueeze(0),
        size=target_size,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    output = base.clone()
    output[(slice(None), *slices)] = resized
    return output


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=min(channels, 8), num_channels=channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(channels, 8), num_channels=channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class CausalRoiRefiner(nn.Module):
    """Lesion-local refiner using high-resolution MRI and coarse region mediator."""

    def __init__(self, in_image_channels: int = 4, num_classes: int = 3, channels: int = 16, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.residual_scale = float(residual_scale)
        input_channels = int(in_image_channels) + int(num_classes) + 3 + 2
        self.stem = nn.Conv3d(input_channels, channels, kernel_size=3, padding=1)
        self.enc = nn.Sequential(_ResidualBlock(channels), _ResidualBlock(channels))
        self.down = nn.Conv3d(channels, channels * 2, kernel_size=3, stride=2, padding=1)
        self.mid = nn.Sequential(_ResidualBlock(channels * 2), _ResidualBlock(channels * 2))
        self.up = nn.ConvTranspose3d(channels * 2, channels, kernel_size=2, stride=2)
        self.dec = nn.Sequential(_ResidualBlock(channels), _ResidualBlock(channels))
        self.out = nn.Conv3d(channels, num_classes, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, roi_image: Tensor, coarse_logits: Tensor) -> Tensor:
        if roi_image.ndim != 5 or coarse_logits.ndim != 5:
            raise ValueError("roi_image and coarse_logits must have shape [B,C,D,H,W].")
        if tuple(roi_image.shape[-3:]) != tuple(coarse_logits.shape[-3:]):
            raise ValueError("roi_image and coarse_logits must have the same spatial shape.")
        coarse_prob = torch.sigmoid(coarse_logits)
        region_prob = subregion_to_region_prob(coarse_prob)
        foreground = coarse_prob.amax(dim=1, keepdim=True)
        uncertainty = (4.0 * coarse_prob * (1.0 - coarse_prob)).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        x = torch.cat([roi_image, coarse_logits, region_prob, foreground, uncertainty], dim=1)
        skip = self.enc(self.stem(x))
        mid = self.mid(self.down(skip))
        up = self.up(mid)
        if tuple(up.shape[-3:]) != tuple(skip.shape[-3:]):
            up = F.interpolate(up, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        residual = self.out(self.dec(up + skip))
        return coarse_logits + self.residual_scale * residual
