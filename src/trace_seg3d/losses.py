"""Losses. One region map (probabilistic OR, paper Eq. 1) is used for training AND evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def subregion_to_region_prob(prob: Tensor) -> Tensor:
    """[B,3(NCR,ED,ET),...] probabilities -> [B,3(WT,TC,ET),...] via probabilistic OR."""

    ncr, ed, et = prob[:, 0], prob[:, 1], prob[:, 2]
    wt = 1 - (1 - ncr) * (1 - ed) * (1 - et)
    tc = 1 - (1 - ncr) * (1 - et)
    return torch.stack([wt, tc, et], dim=1)


def subregion_to_region_target(target: Tensor) -> Tensor:
    ncr, ed, et = target[:, 0], target[:, 1], target[:, 2]
    return torch.stack([torch.maximum(torch.maximum(ncr, ed), et), torch.maximum(ncr, et), et], dim=1)


def batch_soft_dice_loss(prob: Tensor, target: Tensor, eps: float = 1e-5) -> Tensor:
    """nnU-Net style batch Dice: sums over batch + space, so ET-empty cases do not produce 0/0."""

    dims = (0, *range(2, prob.ndim))
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def segmentation_loss(logits: Tensor, target: Tensor, region_weight: float = 1.0) -> Tensor:
    logits = logits.float()
    target = target.float()
    prob = torch.sigmoid(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target) + batch_soft_dice_loss(prob, target)
    if region_weight > 0:
        loss = loss + region_weight * batch_soft_dice_loss(subregion_to_region_prob(prob), subregion_to_region_target(target))
    return loss


def stability_loss(logits: Tensor, transported_logits: Tensor, margin: float, support: float = 0.1) -> Tensor:
    """Eq. 19 restricted to the lesion support: [mean_{v in S} |p_v - p~_v| - m]_+ .

    S = voxels where either the factual or the transported probability exceeds ``support``.
    Averaging over the whole volume (as before) is dominated by background and the hinge
    is almost never active.
    """

    p = torch.sigmoid(logits.float())
    q = torch.sigmoid(transported_logits.float())
    mask = (torch.maximum(p, q) > support).float().detach()
    shift = ((p - q).abs() * mask).sum() / mask.sum().clamp_min(1.0)
    return F.relu(shift - margin)


def multi_ce(logit_list: list[Tensor], labels: Tensor) -> Tensor | None:
    """Sum of CE over proxy fields; labels [B, F] with -100 = unknown. None if nothing is labelled."""

    losses = []
    for i, logits in enumerate(logit_list):
        y = labels[:, i]
        if (y >= 0).any():
            losses.append(F.cross_entropy(logits.float(), y, ignore_index=-100))
    return torch.stack(losses).sum() if losses else None


def orthogonality_loss(z_d: Tensor, z_c: Tensor) -> Tensor:
    return (F.normalize(z_d, dim=1) * F.normalize(z_c, dim=1)).sum(1).pow(2).mean()
