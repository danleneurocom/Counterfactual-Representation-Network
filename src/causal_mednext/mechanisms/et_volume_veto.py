from __future__ import annotations

import torch
from torch import Tensor


def apply_et_volume_veto(
    logits: Tensor,
    region_volume_logits: Tensor | None,
    *,
    num_classes: int,
    region_volume_scale: float,
    et_volume_veto_scale: float,
    et_volume_veto_multiplier: float,
    et_volume_veto_min_fraction: float,
    et_volume_veto_max_bias: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Down-bias implausibly large ET predictions using a learned volume proxy."""

    if (
        et_volume_veto_scale <= 0.0
        or num_classes < 3
        or region_volume_logits is None
        or region_volume_logits.ndim != 2
        or region_volume_logits.shape[1] < 3
    ):
        return logits, {}

    et_prob = torch.sigmoid(logits[:, 2:3].detach())
    predicted_fraction = (et_prob > 0.5).to(dtype=logits.dtype).flatten(1).mean(dim=1)
    proxy_fraction = torch.expm1(region_volume_logits[:, 2].detach()).clamp_min(0.0)
    proxy_fraction = (proxy_fraction / region_volume_scale).clamp(0.0, 1.0)
    allowed_fraction = (proxy_fraction * et_volume_veto_multiplier).clamp(0.0, 1.0)
    if et_volume_veto_min_fraction > 0.0:
        allowed_fraction = allowed_fraction.clamp_min(et_volume_veto_min_fraction)
    denominator = allowed_fraction.clamp_min(max(et_volume_veto_min_fraction, 1e-6))
    relative_excess = ((predicted_fraction - allowed_fraction) / denominator).clamp_min(0.0)
    bias = (et_volume_veto_scale * relative_excess).clamp(max=et_volume_veto_max_bias)
    if bool((bias > 0).any().detach().cpu()):
        logits = logits.clone()
        logits[:, 2] = logits[:, 2] - bias.view(-1, 1, 1, 1)
    return logits, {
        "et_volume_veto_bias": bias,
        "et_volume_veto_predicted_fraction": predicted_fraction,
        "et_volume_veto_allowed_fraction": allowed_fraction,
        "et_volume_proxy_fraction": proxy_fraction,
    }
