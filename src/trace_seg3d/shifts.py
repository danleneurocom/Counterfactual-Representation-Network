"""Controlled, reproducible acquisition shifts applied to a preprocessed (z-scored) test volume.

Used to draw "shift severity -> Dice drop -> audit signal" curves without any new data. Every
shift acts on the image only (labels untouched), is seeded by (case id, shift) so all methods
see exactly the same corrupted volumes, and is followed by the same in-brain z-scoring as the
preprocessing pipeline (background stays 0), i.e. it mimics a different scanner/protocol
*before* our standard normalisation.

    bias      smooth multiplicative bias field (coil inhomogeneity)        strength exp(+-a)
    gamma     per-sequence gamma contrast change (protocol / vendor)        gamma in [1/g, g]
    noise     additive Gaussian noise inside the brain (lower SNR)          sigma in z units
    lowres    thick slices: average-downsample the axial axis, upsample     factor f
"""

from __future__ import annotations

import zlib

import torch
import torch.nn.functional as F
from torch import Tensor

SEVERITY = {
    "bias": {1: 0.15, 2: 0.3, 3: 0.5},
    "gamma": {1: 1.6, 2: 2.5, 3: 3.5},
    "noise": {1: 0.2, 2: 0.4, 3: 0.7},
    "lowres": {1: 3, 2: 5, 3: 7},
}


def parse_shift(text: str | None) -> tuple[str, int] | None:
    if not text or text == "none":
        return None
    name, _, sev = text.partition(":")
    if name not in SEVERITY or int(sev or 1) not in SEVERITY[name]:
        raise ValueError(f"unknown shift {text!r}; use one of {list(SEVERITY)} with severity 1-3, e.g. bias:2")
    return name, int(sev or 1)


def _generator(case_id: str, shift: str, device: torch.device) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(zlib.crc32(f"{case_id}|{shift}".encode()))
    return g


def _zscore(x: Tensor, brain: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    for c in range(x.shape[0]):
        v = x[c][brain]
        out[c][brain] = (v - v.mean()) / v.std().clamp_min(1e-6)
    return out


def apply_shift(image: Tensor, brain: Tensor, case_id: str, shift: tuple[str, int] | None) -> Tensor:
    """image [C,D,H,W] z-scored, brain [D,H,W] bool -> shifted image, same shape and dtype."""

    if shift is None:
        return image
    name, sev = shift
    level = SEVERITY[name][sev]
    x = image.float()
    brain = brain.bool()
    g = _generator(case_id, f"{name}:{sev}", x.device)
    if name == "noise":
        noise = torch.randn(x.shape, generator=g).to(x.device) * level
        x = x + noise * brain
    else:
        # work on non-negative intensities (per sequence, robust min inside the brain)
        lo = torch.stack([torch.quantile(x[c][brain], 0.005) for c in range(x.shape[0])]).view(-1, 1, 1, 1)
        v = (x - lo).clamp_min(0) * brain
        if name == "bias":
            coarse = torch.randn((1, 1, 4, 4, 4), generator=g).to(x.device)
            field = F.interpolate(coarse, size=x.shape[1:], mode="trilinear", align_corners=True)[0, 0]
            field = (field - field.mean()) / field.std().clamp_min(1e-6)
            v = v * torch.exp(level * field)
        elif name == "gamma":
            hi = torch.stack([torch.quantile(v[c][brain], 0.995) for c in range(v.shape[0])]).view(-1, 1, 1, 1).clamp_min(1e-6)
            signs = torch.where(torch.rand(x.shape[0], generator=g) < 0.5, -1.0, 1.0).to(x.device)
            gam = (level ** signs).view(-1, 1, 1, 1)
            v = (v / hi).clamp(0, 1) ** gam * hi
        elif name == "lowres":
            d = x.shape[-1]
            low = F.avg_pool1d(v.reshape(-1, 1, d), kernel_size=level, stride=level, ceil_mode=True)
            v = F.interpolate(low, size=d, mode="linear", align_corners=False).reshape(v.shape)
        x = v
    return _zscore(x, brain).to(image.dtype)


__all__ = ["SEVERITY", "apply_shift", "parse_shift"]
