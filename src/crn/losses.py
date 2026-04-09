from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_cls: float = 1.0
    lambda_seg: float = 0.0
    lambda_rec: float = 0.0
    lambda_dis: float = 0.0
    lambda_adjustment: float = 0.0
    lambda_cf_stability: float = 0.0
    lambda_seg_cf_stability: float = 0.0
    lambda_disease_swap: float = 0.0
    context_stability_margin: float = 0.10
    adjustment_contexts: int | None = None


def as_loss_weights(config: dict | LossWeights | None) -> LossWeights:
    if config is None:
        return LossWeights()
    if isinstance(config, LossWeights):
        return config
    fields = set(LossWeights.__dataclass_fields__)
    return LossWeights(**{key: value for key, value in config.items() if key in fields})


def classification_loss(logits: Tensor, target: Tensor) -> Tensor:
    if target.ndim == 1:
        target = target.unsqueeze(1)
    return F.binary_cross_entropy_with_logits(logits, target.float())


def dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(2, logits.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    return bce + dice_loss(logits, target)


def reconstruction_loss(reconstruction: Tensor, image: Tensor) -> Tensor:
    return F.l1_loss(reconstruction, image)


def decorrelation_loss(z_d: Tensor, z_c: Tensor, eps: float = 1e-6) -> Tensor:
    if z_d.shape[0] < 2:
        return z_d.new_zeros(())
    z_d = z_d - z_d.mean(dim=0, keepdim=True)
    z_c = z_c - z_c.mean(dim=0, keepdim=True)
    z_d = z_d / (z_d.std(dim=0, keepdim=True) + eps)
    z_c = z_c / (z_c.std(dim=0, keepdim=True) + eps)
    corr = z_d.T @ z_c / (z_d.shape[0] - 1)
    return corr.square().mean()


def shifted_permutation(batch_size: int, device: torch.device) -> Tensor:
    if batch_size <= 1:
        return torch.arange(batch_size, device=device)
    return torch.roll(torch.arange(batch_size, device=device), shifts=1)


def bounded_stability_loss(original: Tensor, counterfactual: Tensor, margin: float) -> Tensor:
    delta = (original - counterfactual).abs()
    return torch.relu(delta - margin).mean()


def _context_subset(z_c: Tensor, max_contexts: int | None) -> Tensor:
    if max_contexts is None or max_contexts <= 0 or max_contexts >= z_c.shape[0]:
        return z_c
    return z_c[:max_contexts]


def backdoor_adjusted_logits(model: nn.Module, z_d: Tensor, z_c: Tensor, max_contexts: int | None = None) -> Tensor:
    """Approximate p(y | do(z_d)) by averaging over a context bank from the batch."""

    context_bank = _context_subset(z_c, max_contexts)
    batch_size, num_contexts = z_d.shape[0], context_bank.shape[0]
    z_d_rep = z_d[:, None, :].expand(batch_size, num_contexts, -1).reshape(batch_size * num_contexts, -1)
    z_c_rep = context_bank[None, :, :].expand(batch_size, num_contexts, -1).reshape(batch_size * num_contexts, -1)
    logits = model.predict_from_latents(z_d_rep, z_c_rep)["logits"]
    probs = torch.sigmoid(logits).view(batch_size, num_contexts, -1).mean(dim=1)
    probs = probs.clamp(min=1e-5, max=1.0 - 1e-5)
    return torch.logit(probs)


def compute_crn_losses(
    model: nn.Module,
    batch: dict[str, Tensor],
    outputs: dict[str, Tensor],
    weights: dict | LossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    cfg = as_loss_weights(weights)
    terms: dict[str, Tensor] = {}
    zero = outputs["z_d"].new_zeros(())

    if cfg.lambda_cls and "label" in batch and "logits" in outputs:
        terms["cls"] = classification_loss(outputs["logits"], batch["label"])
    else:
        terms["cls"] = zero

    if cfg.lambda_seg and "mask" in batch and "seg_logits" in outputs:
        terms["seg"] = segmentation_loss(outputs["seg_logits"], batch["mask"])
    else:
        terms["seg"] = zero

    if cfg.lambda_rec and "reconstruction" in outputs:
        terms["rec"] = reconstruction_loss(outputs["reconstruction"], batch["image"])
    else:
        terms["rec"] = zero

    if cfg.lambda_dis:
        terms["dis"] = decorrelation_loss(outputs["z_d"], outputs["z_c"])
    else:
        terms["dis"] = zero

    can_counterfactual = outputs["z_d"].shape[0] > 1
    if cfg.lambda_adjustment and can_counterfactual and "label" in batch and "logits" in outputs:
        adjusted = backdoor_adjusted_logits(model, outputs["z_d"], outputs["z_c"], cfg.adjustment_contexts)
        terms["adjustment"] = classification_loss(adjusted, batch["label"])
    else:
        terms["adjustment"] = zero

    if cfg.lambda_cf_stability and can_counterfactual and "logits" in outputs:
        perm = shifted_permutation(outputs["z_d"].shape[0], outputs["z_d"].device)
        cf_logits = model.predict_from_latents(outputs["z_d"], outputs["z_c"][perm])["logits"]
        terms["cf_stability"] = bounded_stability_loss(
            torch.sigmoid(outputs["logits"]),
            torch.sigmoid(cf_logits),
            cfg.context_stability_margin,
        )
    else:
        terms["cf_stability"] = zero

    if cfg.lambda_seg_cf_stability and can_counterfactual and "seg_logits" in outputs:
        perm = shifted_permutation(outputs["z_d"].shape[0], outputs["z_d"].device)
        cf_seg_logits = model.predict_from_latents(outputs["z_d"], outputs["z_c"][perm])["seg_logits"]
        terms["seg_cf_stability"] = bounded_stability_loss(
            torch.sigmoid(outputs["seg_logits"]),
            torch.sigmoid(cf_seg_logits),
            cfg.context_stability_margin,
        )
    else:
        terms["seg_cf_stability"] = zero

    if cfg.lambda_disease_swap and can_counterfactual and "label" in batch and "logits" in outputs:
        perm = shifted_permutation(outputs["z_d"].shape[0], outputs["z_d"].device)
        donor_logits = model.predict_from_latents(outputs["z_d"][perm], outputs["z_c"])["logits"]
        terms["disease_swap"] = classification_loss(donor_logits, batch["label"][perm])
    else:
        terms["disease_swap"] = zero

    weighted_terms = {
        "cls": cfg.lambda_cls * terms["cls"],
        "seg": cfg.lambda_seg * terms["seg"],
        "rec": cfg.lambda_rec * terms["rec"],
        "dis": cfg.lambda_dis * terms["dis"],
        "adjustment": cfg.lambda_adjustment * terms["adjustment"],
        "cf_stability": cfg.lambda_cf_stability * terms["cf_stability"],
        "seg_cf_stability": cfg.lambda_seg_cf_stability * terms["seg_cf_stability"],
        "disease_swap": cfg.lambda_disease_swap * terms["disease_swap"],
    }
    total = sum(weighted_terms.values(), zero)
    logs = {f"loss/{name}": value.detach() for name, value in terms.items()}
    logs.update({f"weighted/{name}": value.detach() for name, value in weighted_terms.items()})
    logs["loss/total"] = total.detach()
    return total, logs

