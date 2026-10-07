from __future__ import annotations

import torch
from torch import Tensor


def logit(probability: Tensor, eps: float = 1e-4) -> Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


def region_logits_to_subregion_prior(region_logits: Tensor) -> Tensor:
    """Convert WT/TC/ET region logits into NCR/NET, edema, ET logits."""

    region_prob = torch.sigmoid(region_logits)
    wt = region_prob[:, 0:1]
    tc = region_prob[:, 1:2]
    et = region_prob[:, 2:3]
    ncr_net = (tc * (1.0 - et)).clamp(0.0, 1.0)
    edema = (wt * (1.0 - tc)).clamp(0.0, 1.0)
    subregion_prob = torch.cat([ncr_net, edema, et], dim=1)
    return logit(subregion_prob)


def subregion_prob_to_region_prob(subregion_prob: Tensor) -> Tensor:
    """Map NCR/NET, edema, ET probabilities to nested WT/TC/ET regions."""

    ncr_net = subregion_prob[:, 0:1]
    edema = subregion_prob[:, 1:2]
    enhancing = subregion_prob[:, 2:3]
    whole_tumor = 1.0 - (1.0 - ncr_net) * (1.0 - edema) * (1.0 - enhancing)
    tumor_core = 1.0 - (1.0 - ncr_net) * (1.0 - enhancing)
    return torch.cat([whole_tumor, tumor_core, enhancing], dim=1).clamp(0.0, 1.0)


def nested_condition_logits_to_outputs(
    raw_region_logits: Tensor,
    condition_delta: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert WT, TC|WT, ET|TC condition logits into valid BraTS subregions."""

    raw_region_prob = torch.sigmoid(raw_region_logits)
    raw_wt = raw_region_prob[:, 0:1]
    raw_tc = raw_region_prob[:, 1:2]
    raw_et = raw_region_prob[:, 2:3]
    raw_tc_given_wt = (raw_tc / raw_wt.clamp_min(1e-4)).clamp(0.0, 1.0)
    raw_et_given_tc = (raw_et / raw_tc.clamp_min(1e-4)).clamp(0.0, 1.0)
    raw_condition_logits = torch.cat(
        [
            logit(raw_wt),
            logit(raw_tc_given_wt),
            logit(raw_et_given_tc),
        ],
        dim=1,
    )
    condition_logits = raw_condition_logits + condition_delta
    wt_prob = torch.sigmoid(condition_logits[:, 0:1])
    tc_prob = wt_prob * torch.sigmoid(condition_logits[:, 1:2])
    et_prob = tc_prob * torch.sigmoid(condition_logits[:, 2:3])
    region_prob = torch.cat([wt_prob, tc_prob, et_prob], dim=1).clamp(0.0, 1.0)
    ncr_net_prob = (1.0 - (1.0 - tc_prob) / (1.0 - et_prob).clamp_min(1e-4)).clamp(0.0, 1.0)
    edema_prob = (1.0 - (1.0 - wt_prob) / (1.0 - tc_prob).clamp_min(1e-4)).clamp(0.0, 1.0)
    subregion_prob = torch.cat([ncr_net_prob, edema_prob, et_prob], dim=1).clamp(0.0, 1.0)
    return condition_logits, logit(region_prob), logit(subregion_prob)
