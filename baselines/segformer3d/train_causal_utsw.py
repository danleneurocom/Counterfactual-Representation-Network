from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.segformer3d.causal import CausalSegFormer3D, build_causal_segformer3d, default_utsw_scm
from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_utsw import (
    _average_metric_dicts,
    _build_model as _build_baseline_model,
    _case_ids,
    _make_splits,
    _resolve_device,
    _save_json,
)
from crn.metrics import brats_region_metrics


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_dims(dataset: UTSWGliomaDataset) -> dict[str, int]:
    encoder = dataset.metadata_encoder
    if encoder is None:
        return {"context_proxy_dim": 0, "disease_proxy_dim": 0, "annotation_proxy_dim": 0}
    return {
        "context_proxy_dim": int(encoder.context_dim),
        "disease_proxy_dim": int(encoder.disease_dim),
        "annotation_proxy_dim": int(encoder.annotation_dim),
    }


def _metadata_layout(dataset: UTSWGliomaDataset) -> dict[str, list[dict[str, Any]]] | None:
    encoder = dataset.metadata_encoder
    if encoder is None:
        return None
    return encoder.proxy_layout()


def _make_loader(
    root: Path,
    case_ids: list[str],
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = UTSWGliomaDataset(
        root=root,
        volume_size=args.volume_size,
        case_ids=case_ids,
        crop_margin=args.crop_margin,
        prefer_manual_seg=args.prefer_manual_seg,
        use_ants_modalities=args.use_ants_modalities,
        metadata_path=args.metadata_path,
        include_metadata=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def _load_or_make_splits(args: argparse.Namespace, baseline_checkpoint: dict[str, Any], data_root: Path) -> dict[str, list[str]]:
    if args.splits_json:
        return _load_json(Path(args.splits_json))
    if args.limit_cases is not None:
        return _make_splits(_case_ids(data_root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)
    if "splits" in baseline_checkpoint:
        return baseline_checkpoint["splits"]
    return _make_splits(_case_ids(data_root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)


def _build_model_from_dataset(args: argparse.Namespace, dataset: UTSWGliomaDataset) -> CausalSegFormer3D:
    dims = _metadata_dims(dataset)
    return build_causal_segformer3d(
        model_size=args.model_size,
        latent_dim=args.latent_dim,
        num_classes=3,
        **dims,
    )


def _load_baseline_backbone(model: CausalSegFormer3D, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_baseline_state_dict(state_dict, strict_backbone=True)
    return checkpoint


def _require_metadata(dataset: UTSWGliomaDataset, allow_missing: bool) -> None:
    if dataset.metadata_encoder is None and not allow_missing:
        raise FileNotFoundError(
            "Causal SegFormer3D requires UTSW metadata proxies. Pass --metadata-path or "
            "--allow-missing-metadata if you intentionally want a representation-only ablation."
        )


def _set_backbone_trainable(model: CausalSegFormer3D, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


class LightweightAdamW:
    """Minimal AdamW to avoid importing torch._dynamo during optimizer setup."""

    def __init__(
        self,
        parameter_groups: list[dict[str, Any]],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.defaults = {"lr": float(lr), "weight_decay": float(weight_decay), "betas": betas, "eps": float(eps)}
        self.param_groups: list[dict[str, Any]] = []
        for group in parameter_groups:
            params = list(group.get("params", []))
            if not params:
                continue
            item = dict(group)
            item["params"] = params
            item.setdefault("lr", self.defaults["lr"])
            item.setdefault("weight_decay", self.defaults["weight_decay"])
            item.setdefault("betas", self.defaults["betas"])
            item.setdefault("eps", self.defaults["eps"])
            self.param_groups.append(item)
        self.state: dict[Tensor, dict[str, Any]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("LightweightAdamW does not support sparse gradients.")
                state = self.state.setdefault(
                    parameter,
                    {
                        "step": 0,
                        "exp_avg": torch.zeros_like(parameter),
                        "exp_avg_sq": torch.zeros_like(parameter),
                    },
                )
                state["step"] += 1
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** int(state["step"])
                bias_correction2 = 1.0 - beta2 ** int(state["step"])
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
                parameter.addcdiv_(exp_avg, denom, value=-step_size)

    def state_dict(self) -> dict[str, Any]:
        parameter_ids: dict[int, int] = {}
        next_id = 0
        groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            serialized = {key: value for key, value in group.items() if key != "params"}
            ids: list[int] = []
            for parameter in group["params"]:
                key = id(parameter)
                if key not in parameter_ids:
                    parameter_ids[key] = next_id
                    next_id += 1
                ids.append(parameter_ids[key])
            serialized["params"] = ids
            groups.append(serialized)
        state = {
            parameter_ids[id(parameter)]: value
            for parameter, value in self.state.items()
            if id(parameter) in parameter_ids
        }
        return {"state": state, "param_groups": groups, "defaults": self.defaults}


def _build_optimizer(model: CausalSegFormer3D, args: argparse.Namespace) -> LightweightAdamW:
    backbone_lr = float(args.backbone_lr if args.backbone_lr is not None else args.lr)
    causal_lr = float(args.causal_lr if args.causal_lr is not None else args.lr)
    backbone_parameters: list[Tensor] = []
    causal_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_parameters.append(parameter)
        else:
            causal_parameters.append(parameter)

    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": backbone_lr, "name": "segformer3d_backbone"})
    if causal_parameters:
        parameter_groups.append({"params": causal_parameters, "lr": causal_lr, "name": "causal_heads"})
    return LightweightAdamW(parameter_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))


def _build_teacher_model(
    checkpoint_path: str | None,
    dataset: UTSWGliomaDataset,
    device: torch.device,
) -> nn.Module | None:
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    state_dict = checkpoint.get("model", checkpoint)
    if "latent_dim" in config or "proxy_dims" in checkpoint:
        dims = checkpoint.get("proxy_dims") or _metadata_dims(dataset)
        teacher = build_causal_segformer3d(
            model_size=str(config.get("model_size", "base")),
            latent_dim=int(config.get("latent_dim", 128)),
            num_classes=3,
            **dims,
        )
    else:
        teacher = _build_baseline_model(str(config.get("model_size", "base")), num_classes=3)
    teacher.load_state_dict(state_dict, strict=True)
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


@torch.no_grad()
def _teacher_logits(teacher: nn.Module | None, image: Tensor) -> Tensor | None:
    if teacher is None:
        return None
    output = teacher(image)
    if isinstance(output, dict):
        logits = output["logits"]
    else:
        logits = output
    if not isinstance(logits, Tensor):
        raise TypeError("Teacher model must return logits tensor or a dict containing 'logits'.")
    return logits.detach()


@torch.no_grad()
def _subsample_context_bank(bank: Tensor, max_contexts: int, strategy: str, seed: int) -> Tensor:
    if max_contexts <= 0 or bank.shape[0] <= max_contexts:
        return bank
    if strategy == "uniform":
        positions = torch.linspace(0, bank.shape[0] - 1, steps=max_contexts)
        return bank[positions.round().long()]
    if strategy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.randperm(bank.shape[0], generator=generator)[:max_contexts]
        return bank[indices.sort().values]
    if strategy == "farthest":
        selected = [0]
        distances = torch.cdist(bank[:1], bank).squeeze(0)
        while len(selected) < max_contexts:
            next_index = int(distances.argmax().item())
            selected.append(next_index)
            next_distances = torch.cdist(bank[next_index : next_index + 1], bank).squeeze(0)
            distances = torch.minimum(distances, next_distances)
            distances[selected] = -1.0
        return bank[torch.tensor(selected, dtype=torch.long)]
    raise ValueError(f"Unknown context bank sampling strategy: {strategy}")


@torch.no_grad()
def build_context_bank(
    model: CausalSegFormer3D,
    loader: DataLoader,
    device: torch.device,
    max_contexts: int,
    max_batches: int | None = None,
    sampling: str = "uniform",
    seed: int = 7,
) -> Tensor | None:
    model.eval()
    chunks: list[Tensor] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="context-bank", leave=False), start=1):
        image = batch["image"].to(device)
        features = model.encode_features(image)
        _, z_c = model.encode_latents(features)
        chunks.append(z_c.detach().cpu())
        if max_batches is not None and batch_idx >= max_batches:
            break
    if not chunks:
        return None
    bank = torch.cat(chunks, dim=0)
    return _subsample_context_bank(bank, max_contexts=max_contexts, strategy=sampling, seed=seed)


def _proxy_mse(prediction: Tensor | None, target: Tensor | None) -> Tensor | None:
    if prediction is None or target is None:
        return None
    return F.mse_loss(prediction, target.to(device=prediction.device, dtype=prediction.dtype))


def _parse_weights(spec: str | None, expected: int, name: str) -> tuple[float, ...]:
    if spec is None:
        return tuple(1.0 for _ in range(expected))
    values = tuple(float(part.strip()) for part in spec.split(",") if part.strip())
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated floats, got {spec!r}")
    return values


def _channel_weighted_dice_loss(logits: Tensor, target: Tensor, weights: tuple[float, ...], eps: float = 1e-6) -> Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    channel_weights = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    weighted = (1.0 - dice) * channel_weights.view(1, -1)
    return weighted.sum(dim=1).div(channel_weights.sum().clamp_min(eps)).mean()


def _channel_weighted_bce_loss(logits: Tensor, target: Tensor, weights: tuple[float, ...]) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    channel_weights = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    shape = (1, len(weights)) + (1,) * (loss.ndim - 2)
    return (loss * channel_weights.view(shape)).mean()


def _region_probabilities(values: Tensor) -> Tensor:
    ncr_net = values[:, 0]
    edema = values[:, 1]
    enhancing = values[:, 2]
    whole_tumor = 1.0 - (1.0 - ncr_net) * (1.0 - edema) * (1.0 - enhancing)
    tumor_core = 1.0 - (1.0 - ncr_net) * (1.0 - enhancing)
    return torch.stack([whole_tumor, tumor_core, enhancing], dim=1)


def _region_weighted_dice_loss(logits: Tensor, target: Tensor, weights: tuple[float, ...], eps: float = 1e-6) -> Tensor:
    probs = _region_probabilities(torch.sigmoid(logits))
    region_target = _region_probabilities(target).clamp(0.0, 1.0)
    dims = tuple(range(2, probs.ndim))
    region_weights = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    intersection = (probs * region_target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + region_target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    weighted = (1.0 - dice) * region_weights.view(1, -1)
    return weighted.sum(dim=1).div(region_weights.sum().clamp_min(eps)).mean()


def _segmentation_terms(logits: Tensor, target: Tensor, args: argparse.Namespace, prefix: str) -> dict[str, Tensor]:
    channel_weights = _parse_weights(args.channel_loss_weights, 3, "--channel-loss-weights")
    region_weights = _parse_weights(args.region_loss_weights, 3, "--region-loss-weights")
    terms = {
        prefix: _channel_weighted_bce_loss(logits, target, channel_weights)
        + _channel_weighted_dice_loss(logits, target, channel_weights)
    }
    if args.lambda_region_loss > 0.0:
        name = "region" if prefix == "seg" else f"{prefix}_region"
        terms[name] = _region_weighted_dice_loss(logits, target, region_weights)
    return terms


def _probability_distillation_loss(student_logits: Tensor, teacher_logits: Tensor, args: argparse.Namespace) -> Tensor:
    weights = _parse_weights(args.distill_channel_weights, 3, "--distill-channel-weights")
    temperature = max(float(args.teacher_temperature), 1e-6)
    student = torch.sigmoid(student_logits / temperature)
    teacher = torch.sigmoid(teacher_logits.detach() / temperature)
    loss = (student - teacher).pow(2)
    channel_weights = torch.as_tensor(weights, device=student_logits.device, dtype=student_logits.dtype)
    shape = (1, len(weights)) + (1,) * (loss.ndim - 2)
    return (loss * channel_weights.view(shape)).mean()


def _probability_consistency_loss(student_logits: Tensor, reference_logits: Tensor, args: argparse.Namespace) -> Tensor:
    weights = _parse_weights(args.distill_channel_weights, 3, "--distill-channel-weights")
    student = torch.sigmoid(student_logits)
    reference = torch.sigmoid(reference_logits.detach())
    loss = (student - reference).pow(2)
    channel_weights = torch.as_tensor(weights, device=student_logits.device, dtype=student_logits.dtype)
    shape = (1, len(weights)) + (1,) * (loss.ndim - 2)
    return (loss * channel_weights.view(shape)).mean()


def _typed_proxy_loss(
    prediction: Tensor | None,
    target: Tensor | None,
    layout: list[dict[str, Any]] | None,
) -> Tensor | None:
    if prediction is None or target is None:
        return None
    if not layout:
        return _proxy_mse(prediction, target)
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    losses: list[Tensor] = []
    for spec in layout:
        start = int(spec["start"])
        end = int(spec["end"])
        if end <= start:
            continue
        pred_slice = prediction[:, start:end]
        target_slice = target[:, start:end]
        if spec.get("kind") == "categorical" and end - start > 1:
            class_target = target_slice.argmax(dim=1)
            losses.append(F.cross_entropy(pred_slice, class_target))
        else:
            losses.append(F.mse_loss(pred_slice, target_slice))
    if not losses:
        return None
    return torch.stack(losses).mean()


def _proxy_loss(
    prediction: Tensor | None,
    target: Tensor | None,
    layout: list[dict[str, Any]] | None,
    mode: str,
) -> Tensor | None:
    if mode == "mse":
        return _proxy_mse(prediction, target)
    if mode == "typed":
        return _typed_proxy_loss(prediction, target, layout)
    raise ValueError(f"Unknown proxy loss mode: {mode}")


def _should_compute_context_swap(args: argparse.Namespace) -> bool:
    if args.context_swap_strategy == "none":
        return False
    return any(
        value > 0.0
        for value in (
            args.lambda_context_swap,
            args.lambda_context_swap_region,
            args.lambda_context_swap_consistency,
            args.lambda_context_swap_teacher_distill,
        )
    )


def _select_swap_contexts(z_c: Tensor, context_bank: Tensor, args: argparse.Namespace) -> Tensor:
    bank = context_bank.to(device=z_c.device, dtype=z_c.dtype)
    if bank.ndim != 2 or bank.shape[1] != z_c.shape[1]:
        raise ValueError(f"context_bank must have shape [K, {z_c.shape[1]}], got {tuple(bank.shape)}")
    if args.context_swap_strategy == "random":
        indices = torch.randint(0, bank.shape[0], (z_c.shape[0],), device=z_c.device)
    elif args.context_swap_strategy == "farthest":
        distances = torch.cdist(z_c.detach(), bank)
        indices = distances.argmax(dim=1)
    elif args.context_swap_strategy == "nearest":
        distances = torch.cdist(z_c.detach(), bank)
        indices = distances.argmin(dim=1)
    else:
        raise ValueError(f"Unknown context swap strategy: {args.context_swap_strategy}")
    return bank[indices]


def _add_context_swap_outputs(
    model: CausalSegFormer3D,
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    context_bank: Tensor | None,
    args: argparse.Namespace,
) -> None:
    if context_bank is None or not _should_compute_context_swap(args):
        return
    features = outputs["features"]
    z_d = outputs["z_d"]
    z_c = outputs["z_c"]
    if not isinstance(features, tuple) or not isinstance(z_d, Tensor) or not isinstance(z_c, Tensor):
        return
    swap_z_c = _select_swap_contexts(z_c, context_bank, args)
    outputs["context_swap_logits"] = model.segment_from_latents(features, z_d, swap_z_c)


def _orthogonality_loss(z_d: Tensor, z_c: Tensor) -> Tensor:
    z_d_norm = F.normalize(z_d, dim=1)
    z_c_norm = F.normalize(z_c, dim=1)
    return (z_d_norm * z_c_norm).sum(dim=1).pow(2).mean()


def _bounded_context_shift_loss(factual_logits: Tensor, adjusted_logits: Tensor, margin: float) -> Tensor:
    shift = (torch.sigmoid(factual_logits) - torch.sigmoid(adjusted_logits)).abs().mean()
    return torch.relu(shift - float(margin))


def _context_response_loss(
    factual_logits: Tensor,
    adjusted_logits: Tensor,
    target_shift: float,
    target: Tensor | None = None,
) -> Tensor:
    factual_prob = torch.sigmoid(factual_logits)
    shift_map = (factual_prob - torch.sigmoid(adjusted_logits)).abs()
    if target is not None:
        target = target.to(device=factual_logits.device, dtype=factual_logits.dtype)
        if tuple(target.shape[-3:]) != tuple(factual_logits.shape[-3:]):
            target = F.interpolate(target, size=factual_logits.shape[-3:], mode="nearest")
        predicted_focus = (factual_prob.detach() > 0.20).to(dtype=factual_logits.dtype)
        lesion_focus = torch.maximum(target.clamp(0.0, 1.0), predicted_focus)
        uncertainty = (4.0 * factual_prob.detach() * (1.0 - factual_prob.detach())).clamp(0.0, 1.0)
        weights = lesion_focus + 0.25 * uncertainty
        shift = (shift_map * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        shift = shift_map.mean()
    target_value = torch.as_tensor(float(target_shift), device=shift.device, dtype=shift.dtype)
    scale = target_value.abs().clamp_min(1e-6)
    return ((shift - target_value) / scale).pow(2)


def _region_volume_target(target: Tensor, scale: float = 1000.0) -> Tensor:
    fractions = _region_probabilities(target).clamp(0.0, 1.0).mean(dim=(2, 3, 4))
    return torch.log1p(float(scale) * fractions)


def _region_volume_proxy_loss(prediction: Tensor | None, target: Tensor, scale: float = 1000.0) -> Tensor | None:
    if prediction is None:
        return None
    region_target = _region_volume_target(target, scale=scale).to(device=prediction.device, dtype=prediction.dtype)
    return F.smooth_l1_loss(prediction, region_target)


def _binary_dice_loss_from_logits(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _spatial_disease_attention_loss(prediction: Tensor | None, target: Tensor) -> Tensor | None:
    if prediction is None:
        return None
    wt_target = _region_probabilities(target).clamp(0.0, 1.0)[:, :1]
    wt_target = wt_target.to(device=prediction.device, dtype=prediction.dtype)
    return F.binary_cross_entropy_with_logits(prediction, wt_target) + _binary_dice_loss_from_logits(prediction, wt_target)


def _spatial_region_head_loss(prediction: Tensor | None, target: Tensor, args: argparse.Namespace) -> Tensor | None:
    if prediction is None:
        return None
    region_target = _region_probabilities(target).clamp(0.0, 1.0).to(device=prediction.device, dtype=prediction.dtype)
    weights = _parse_weights(args.region_loss_weights, 3, "--region-loss-weights")
    return _channel_weighted_bce_loss(prediction, region_target, weights) + _channel_weighted_dice_loss(
        prediction,
        region_target,
        weights,
    )


def _subregion_prior_loss(prediction: Tensor | None, target: Tensor, args: argparse.Namespace) -> Tensor | None:
    if prediction is None:
        return None
    weights = _parse_weights(args.channel_loss_weights, 3, "--channel-loss-weights")
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    return _channel_weighted_bce_loss(prediction, target, weights) + _channel_weighted_dice_loss(prediction, target, weights)


def _voxel_balanced_bce_loss(logits: Tensor, target: Tensor, max_pos_weight: float = 50.0) -> Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    dims = tuple(idx for idx in range(target.ndim) if idx != 1)
    positives = target.sum(dim=dims).clamp_min(1.0)
    negatives = (1.0 - target).sum(dim=dims).clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(1.0, float(max_pos_weight))
    view_shape = (1, target.shape[1]) + (1,) * (target.ndim - 2)
    weights = torch.where(target > 0.5, pos_weight.view(view_shape), torch.ones_like(target))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weights).sum() / weights.sum().clamp_min(1.0)


def _focal_tversky_loss_from_logits(
    logits: Tensor,
    target: Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, logits.ndim))
    true_pos = (probs * target).sum(dim=dims)
    false_neg = ((1.0 - probs) * target).sum(dim=dims)
    false_pos = (probs * (1.0 - target)).sum(dim=dims)
    tversky = (true_pos + eps) / (true_pos + float(alpha) * false_neg + float(beta) * false_pos + eps)
    # Fractional powers have an infinite derivative at exactly zero, which can
    # produce NaN gradients for confident/perfect mediator predictions.
    return (1.0 - tversky).clamp_min(eps).pow(float(gamma)).mean()


def _balanced_region_mediator_loss(prediction: Tensor | None, target: Tensor) -> Tensor | None:
    if prediction is None:
        return None
    region_target = _region_probabilities(target).clamp(0.0, 1.0).to(device=prediction.device, dtype=prediction.dtype)
    return _voxel_balanced_bce_loss(prediction, region_target) + _focal_tversky_loss_from_logits(
        prediction,
        region_target,
    )


def _balanced_subregion_mediator_loss(prediction: Tensor | None, target: Tensor) -> Tensor | None:
    if prediction is None:
        return None
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    return _voxel_balanced_bce_loss(prediction, target) + _focal_tversky_loss_from_logits(prediction, target)


def _subregion_class_target(target: Tensor) -> Tensor:
    target = target.detach()
    labels = torch.zeros(target.shape[0], *target.shape[2:], device=target.device, dtype=torch.long)
    labels = torch.where(target[:, 0] > 0.5, torch.ones_like(labels), labels)
    labels = torch.where(target[:, 1] > 0.5, torch.full_like(labels, 2), labels)
    labels = torch.where(target[:, 2] > 0.5, torch.full_like(labels, 3), labels)
    return labels


def _balanced_class_weights(labels: Tensor, num_classes: int) -> Tensor:
    counts = torch.bincount(labels.reshape(-1), minlength=num_classes).to(dtype=torch.float32, device=labels.device)
    present = counts > 0
    weights = torch.ones(num_classes, device=labels.device, dtype=torch.float32)
    if present.any():
        weights[present] = counts[present].sum() / (present.sum().float() * counts[present].clamp_min(1.0))
        weights = weights / weights[present].mean().clamp_min(1e-6)
    return weights.clamp(0.05, 20.0)


def _prototype_mediator_loss(
    prototype_logits: Tensor | None,
    subregion_logits: Tensor | None,
    target: Tensor,
    args: argparse.Namespace,
) -> Tensor | None:
    if prototype_logits is None:
        return None
    if tuple(prototype_logits.shape[-3:]) != tuple(target.shape[-3:]):
        prototype_logits = F.interpolate(prototype_logits, size=target.shape[-3:], mode="trilinear", align_corners=False)
    labels = _subregion_class_target(target).to(device=prototype_logits.device)
    class_weights = _balanced_class_weights(labels, prototype_logits.shape[1]).to(
        device=prototype_logits.device,
        dtype=prototype_logits.dtype,
    )
    ce = F.cross_entropy(prototype_logits, labels, weight=class_weights)
    if subregion_logits is None:
        subregion_logits = prototype_logits[:, 1:] - prototype_logits[:, 0:1]
    seg = _subregion_prior_loss(subregion_logits, target, args)
    return ce if seg is None else ce + seg


def _boundary_target_from_subregions(target: Tensor, kernel_size: int = 3) -> Tensor:
    foreground = target.amax(dim=1, keepdim=True).float()
    padding = int(kernel_size) // 2
    dilated = F.max_pool3d(foreground, kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool3d(1.0 - foreground, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def _boundary_mediator_loss(prediction: Tensor | None, target: Tensor) -> Tensor | None:
    if prediction is None:
        return None
    boundary_target = _boundary_target_from_subregions(target).to(device=prediction.device, dtype=prediction.dtype)
    if tuple(prediction.shape[-3:]) != tuple(boundary_target.shape[-3:]):
        prediction = F.interpolate(prediction, size=boundary_target.shape[-3:], mode="trilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(prediction, boundary_target) + _binary_dice_loss_from_logits(
        prediction,
        boundary_target,
    )


def _causal_refiner_sparsity_loss(delta: Tensor | None, attention_logits: Tensor | None) -> Tensor | None:
    if delta is None:
        return None
    if attention_logits is None:
        return delta.abs().mean()
    attention = torch.sigmoid(attention_logits).to(device=delta.device, dtype=delta.dtype)
    outside = (1.0 - attention).clamp(0.0, 1.0)
    return (delta.abs() * outside).mean()


def _cascade_error_correction_loss(
    cascade_logits: Tensor | None,
    base_logits: Tensor | None,
    target: Tensor,
    args: argparse.Namespace,
) -> Tensor | None:
    if cascade_logits is None or base_logits is None:
        return None
    target = target.to(device=cascade_logits.device, dtype=cascade_logits.dtype)
    base_prob = torch.sigmoid(base_logits.detach())
    false_negative_pressure = target * (1.0 - base_prob)
    false_positive_pressure = (1.0 - target) * base_prob
    uncertainty = (4.0 * base_prob * (1.0 - base_prob)).clamp(0.0, 1.0)
    weights = 1.0 + 3.0 * false_negative_pressure + 2.0 * false_positive_pressure + uncertainty
    bce = F.binary_cross_entropy_with_logits(cascade_logits, target, reduction="none")
    weighted_bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
    dice = _channel_weighted_dice_loss(
        cascade_logits,
        target,
        _parse_weights(args.channel_loss_weights, 3, "--channel-loss-weights"),
    )
    return weighted_bce + dice


def _nested_causal_router_advantage_loss(
    router_logits: Tensor | None,
    base_logits: Tensor | None,
    nested_logits: Tensor | None,
    target: Tensor,
    margin: float = 0.02,
) -> Tensor | None:
    if router_logits is None or base_logits is None or nested_logits is None:
        return None
    target = target.to(device=router_logits.device, dtype=router_logits.dtype)
    base_logits = base_logits.to(device=router_logits.device, dtype=router_logits.dtype)
    nested_logits = nested_logits.to(device=router_logits.device, dtype=router_logits.dtype)
    with torch.no_grad():
        base_error = (torch.sigmoid(base_logits) - target).abs()
        nested_error = (torch.sigmoid(nested_logits) - target).abs()
        advantage = base_error - nested_error
        router_target = (advantage > float(margin)).to(dtype=router_logits.dtype)
        lesion_weight = 1.0 + 2.0 * target
        advantage_weight = 1.0 + 2.0 * (advantage.abs() > float(margin)).to(dtype=router_logits.dtype)
        weights = lesion_weight * advantage_weight
    loss = F.binary_cross_entropy_with_logits(router_logits, router_target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _frontdoor_router_advantage_loss(
    router_logits: Tensor | None,
    base_logits: Tensor | None,
    frontdoor_logits: Tensor | None,
    target: Tensor,
    margin: float = 0.02,
) -> Tensor | None:
    if router_logits is None or base_logits is None or frontdoor_logits is None:
        return None
    target = target.to(device=router_logits.device, dtype=router_logits.dtype)
    base_logits = base_logits.to(device=router_logits.device, dtype=router_logits.dtype)
    frontdoor_logits = frontdoor_logits.to(device=router_logits.device, dtype=router_logits.dtype)
    if tuple(target.shape[-3:]) != tuple(router_logits.shape[-3:]):
        target = F.interpolate(target, size=router_logits.shape[-3:], mode="nearest")
    with torch.no_grad():
        base_prob = torch.sigmoid(base_logits)
        mediator_prob = torch.sigmoid(frontdoor_logits)
        base_error = (base_prob - target).abs()
        mediator_error = (mediator_prob - target).abs()
        advantage = base_error - mediator_error
        router_target = (advantage > float(margin)).to(dtype=router_logits.dtype)
        lesion_weight = 1.0 + 2.0 * target
        uncertainty_weight = 1.0 + (4.0 * base_prob * (1.0 - base_prob)).clamp(0.0, 1.0)
        advantage_weight = 1.0 + 2.0 * (advantage.abs() > float(margin)).to(dtype=router_logits.dtype)
        weights = lesion_weight * uncertainty_weight * advantage_weight
    loss = F.binary_cross_entropy_with_logits(router_logits, router_target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _sdd_distillation_loss(student: Tensor | None, teacher: Tensor | None) -> Tensor | None:
    if student is None or teacher is None:
        return None
    return F.smooth_l1_loss(student, teacher.detach().to(device=student.device, dtype=student.dtype))


def _kl_from_logits(source_logits: Tensor, target_logits: Tensor) -> Tensor:
    source_prob = torch.softmax(source_logits, dim=1)
    source_log_prob = torch.log_softmax(source_logits, dim=1)
    target_log_prob = torch.log_softmax(target_logits, dim=1)
    return (source_prob * (source_log_prob - target_log_prob)).sum(dim=1).mean()


def _symmetric_kl_from_logits(left_logits: Tensor, right_logits: Tensor) -> Tensor:
    return 0.5 * (_kl_from_logits(left_logits, right_logits) + _kl_from_logits(right_logits, left_logits))


def _observed_treatment_label(batch: dict[str, Any], device: torch.device) -> Tensor | None:
    label = batch.get("observed_treatment_label")
    if isinstance(label, Tensor):
        return label.to(device=device).view(-1).long()
    treatment = batch.get("observed_treatment")
    if isinstance(treatment, Tensor):
        treatment = treatment.to(device=device)
        if treatment.ndim == 2 and treatment.shape[1] > 1:
            return treatment.argmax(dim=1).long()
        return (treatment.view(-1) > 0.5).long()
    return None


def _sdd_treatment_terms(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    batch: dict[str, Any],
) -> dict[str, Tensor]:
    joint = outputs.get("sdd_treatment_joint_logits")
    z_logits = outputs.get("sdd_treatment_z_logits")
    c_logits = outputs.get("sdd_treatment_c_logits")
    if not isinstance(joint, Tensor) or not isinstance(z_logits, Tensor) or not isinstance(c_logits, Tensor):
        return {}
    label = _observed_treatment_label(batch, joint.device)
    if label is None:
        return {}
    label = label[: joint.shape[0]]
    return {
        "sdd_treatment": F.cross_entropy(joint, label),
        "sdd_treatment_disentangle": F.cross_entropy(z_logits, label)
        + F.cross_entropy(c_logits, label)
        + _symmetric_kl_from_logits(c_logits, z_logits)
        + _kl_from_logits(joint, c_logits)
        + _kl_from_logits(joint, z_logits),
    }


def _sdd_outcome_terms(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    target: Tensor,
    scale: float,
) -> dict[str, Tensor]:
    joint = outputs.get("sdd_outcome_joint_logits")
    y_logits = outputs.get("sdd_outcome_y_logits")
    c_logits = outputs.get("sdd_outcome_c_logits")
    if not isinstance(joint, Tensor) or not isinstance(y_logits, Tensor) or not isinstance(c_logits, Tensor):
        return {}
    region_target = _region_volume_target(target, scale=scale).to(device=joint.device, dtype=joint.dtype)
    return {
        "sdd_outcome": F.smooth_l1_loss(joint, region_target),
        "sdd_outcome_disentangle": F.smooth_l1_loss(y_logits, region_target)
        + F.smooth_l1_loss(c_logits, region_target)
        + _symmetric_kl_from_logits(c_logits, y_logits)
        + _kl_from_logits(joint, c_logits)
        + _kl_from_logits(joint, y_logits),
    }


def _linear_mmd_by_treatment(rep: Tensor, treatment_label: Tensor) -> Tensor:
    treatment_label = treatment_label.to(device=rep.device).view(-1).long()
    count = min(rep.shape[0], treatment_label.shape[0])
    rep = rep[:count]
    treatment_label = treatment_label[:count]
    treated = rep[treatment_label > 0]
    control = rep[treatment_label <= 0]
    if treated.numel() == 0 or control.numel() == 0:
        return rep.sum() * 0.0
    return (treated.mean(dim=0) - control.mean(dim=0)).pow(2).mean()


def _sdd_imbalance_term(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    batch: dict[str, Any],
) -> Tensor | None:
    z_d = outputs.get("z_d")
    if not isinstance(z_d, Tensor):
        return None
    label = _observed_treatment_label(batch, z_d.device)
    if label is None:
        return None
    reps = [z_d]
    labels = [label]
    bank_z_d = outputs.get("sdd_bank_z_d")
    bank_label = outputs.get("sdd_bank_treatment_label")
    if isinstance(bank_z_d, Tensor) and isinstance(bank_label, Tensor):
        reps.append(bank_z_d.to(device=z_d.device, dtype=z_d.dtype))
        labels.append(bank_label.to(device=z_d.device).view(-1).long())
    return _linear_mmd_by_treatment(torch.cat(reps, dim=0), torch.cat(labels, dim=0))


def _cite_contrastive_loss(
    anchor: Tensor | None,
    positive: Tensor | None,
    negative: Tensor | None,
    bank_negative: Tensor | None,
    temperature: float,
) -> Tensor | None:
    if anchor is None or positive is None or negative is None:
        return None
    if anchor.ndim != 2 or positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("CITE contrastive tensors must have shape [N, D].")
    if positive.shape[0] == 1 and anchor.shape[0] > 1:
        positive = positive.expand(anchor.shape[0], -1)
    if anchor.shape != positive.shape or negative.shape[1] != anchor.shape[1]:
        raise ValueError(
            "CITE contrastive anchor/positive must match and negatives must share feature dim, "
            f"got {tuple(anchor.shape)}, {tuple(positive.shape)}, {tuple(negative.shape)}"
        )
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)
    positive_logits = (anchor * positive).sum(dim=1, keepdim=True)
    negative_logits = anchor @ negative.T
    if bank_negative is not None:
        if bank_negative.ndim != 2 or bank_negative.shape[1] != anchor.shape[1]:
            raise ValueError(
                "CITE bank negatives must have shape [K, D] with the same D as anchor, "
                f"got {tuple(bank_negative.shape)} and anchor {tuple(anchor.shape)}"
            )
        bank_negative = F.normalize(bank_negative, dim=1)
        negative_logits = torch.cat([negative_logits, anchor @ bank_negative.T], dim=1)
    logits = torch.cat([positive_logits, negative_logits], dim=1) / max(float(temperature), 1e-6)
    labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


def _causal_loss_terms(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    batch: dict[str, Any],
    args: argparse.Namespace,
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Tensor]:
    target = batch["mask"].to(outputs["logits"].device)
    terms: dict[str, Tensor] = _segmentation_terms(outputs["logits"], target, args, "seg")
    terms["orthogonal"] = _orthogonality_loss(outputs["z_d"], outputs["z_c"])
    if "adjusted_logits" in outputs:
        terms.update(_segmentation_terms(outputs["adjusted_logits"], target, args, "adjustment"))
        terms["context_stability"] = _bounded_context_shift_loss(
            outputs["logits"],
            outputs["adjusted_logits"],
            margin=args.context_stability_margin,
        )
        if args.lambda_context_response > 0.0 and args.context_response_target > 0.0:
            terms["context_response"] = _context_response_loss(
                outputs["logits"],
                outputs["adjusted_logits"],
                target_shift=args.context_response_target,
                target=target,
            )

    if "context_swap_logits" in outputs:
        channel_weights = _parse_weights(args.channel_loss_weights, 3, "--channel-loss-weights")
        region_weights = _parse_weights(args.region_loss_weights, 3, "--region-loss-weights")
        terms["context_swap"] = _channel_weighted_bce_loss(outputs["context_swap_logits"], target, channel_weights) + _channel_weighted_dice_loss(
            outputs["context_swap_logits"],
            target,
            channel_weights,
        )
        if args.lambda_context_swap_region > 0.0:
            terms["context_swap_region"] = _region_weighted_dice_loss(
                outputs["context_swap_logits"],
                target,
                region_weights,
            )
        if args.lambda_context_swap_consistency > 0.0:
            terms["context_swap_consistency"] = _probability_consistency_loss(
                outputs["context_swap_logits"],
                outputs["logits"],
                args,
            )

    teacher_logits = outputs.get("teacher_logits")
    if isinstance(teacher_logits, Tensor):
        if args.lambda_teacher_distill > 0.0:
            terms["teacher_distill"] = _probability_distillation_loss(outputs["logits"], teacher_logits, args)
        if "adjusted_logits" in outputs and args.lambda_adjusted_teacher_distill > 0.0:
            terms["adjusted_teacher_distill"] = _probability_distillation_loss(
                outputs["adjusted_logits"],
                teacher_logits,
                args,
            )
        if "context_swap_logits" in outputs and args.lambda_context_swap_teacher_distill > 0.0:
            terms["context_swap_teacher_distill"] = _probability_distillation_loss(
                outputs["context_swap_logits"],
                teacher_logits,
                args,
            )
        if "region_causal_logits" in outputs and getattr(args, "lambda_region_causal_teacher_distill", 0.0) > 0.0:
            terms["region_causal_teacher_distill"] = _probability_distillation_loss(
                outputs["region_causal_logits"],
                teacher_logits,
                args,
            )

    proxy_terms = {
        "context_proxy": _proxy_loss(
            outputs.get("context_proxy_logits"),
            batch.get("observed_context"),
            None if proxy_layout is None else proxy_layout.get("context"),
            args.proxy_loss_mode,
        ),
        "disease_proxy": _proxy_loss(
            outputs.get("disease_proxy_logits"),
            batch.get("observed_disease"),
            None if proxy_layout is None else proxy_layout.get("disease"),
            args.proxy_loss_mode,
        ),
        "annotation_proxy": _proxy_loss(
            outputs.get("annotation_proxy_logits"),
            batch.get("observed_annotation"),
            None if proxy_layout is None else proxy_layout.get("annotation"),
            args.proxy_loss_mode,
        ),
        "context_from_disease_adversary": _proxy_loss(
            outputs.get("context_from_disease_logits"),
            batch.get("observed_context"),
            None if proxy_layout is None else proxy_layout.get("context"),
            args.proxy_loss_mode,
        ),
        "disease_from_context_adversary": _proxy_loss(
            outputs.get("disease_from_context_logits"),
            batch.get("observed_disease"),
            None if proxy_layout is None else proxy_layout.get("disease"),
            args.proxy_loss_mode,
        ),
        "region_volume_proxy": _region_volume_proxy_loss(
            outputs.get("region_volume_logits"),
            target,
            scale=getattr(args, "region_volume_scale", 1000.0),
        ),
        "region_from_context_adversary": _region_volume_proxy_loss(
            outputs.get("region_from_context_logits"),
            target,
            scale=getattr(args, "region_volume_scale", 1000.0),
        ),
        "sdd_context_teacher": _proxy_loss(
            outputs.get("sdd_context_teacher_logits"),
            batch.get("observed_context"),
            None if proxy_layout is None else proxy_layout.get("context"),
            args.proxy_loss_mode,
        ),
        "sdd_region_teacher": _region_volume_proxy_loss(
            outputs.get("sdd_region_teacher_logits"),
            target,
            scale=getattr(args, "region_volume_scale", 1000.0),
        ),
        "sdd_context_distill": _sdd_distillation_loss(
            outputs.get("context_proxy_logits"),
            outputs.get("sdd_context_teacher_logits"),
        ),
        "sdd_region_distill": _sdd_distillation_loss(
            outputs.get("region_volume_logits"),
            outputs.get("sdd_region_teacher_logits"),
        ),
        "spatial_disease_attention": _spatial_disease_attention_loss(
            outputs.get("disease_attention_logits"),
            target,
        ),
        "spatial_region_head": _spatial_region_head_loss(
            outputs.get("spatial_region_logits"),
            target,
            args,
        ),
        "subregion_prior": _subregion_prior_loss(
            outputs.get("subregion_prior_logits"),
            target,
            args,
        ),
        "prototype_mediator": _prototype_mediator_loss(
            outputs.get("prototype_logits"),
            outputs.get("prototype_subregion_logits"),
            target,
            args,
        ),
        "category_confounder": _subregion_prior_loss(
            outputs.get("category_confounder_logits"),
            target,
            args,
        ),
        "modality_prior": _subregion_prior_loss(
            outputs.get("modality_prior_logits"),
            target,
            args,
        ),
        "frontdoor_region": _spatial_region_head_loss(
            outputs.get("frontdoor_region_logits"),
            target,
            args,
        ),
        "frontdoor_subregion": _subregion_prior_loss(
            outputs.get("frontdoor_subregion_logits"),
            target,
            args,
        ),
        "frontdoor_logits": _subregion_prior_loss(
            outputs.get("frontdoor_logits"),
            target,
            args,
        ),
        "frontdoor_balanced_region": _balanced_region_mediator_loss(
            outputs.get("frontdoor_region_logits"),
            target,
        ),
        "frontdoor_balanced_subregion": _balanced_subregion_mediator_loss(
            outputs.get("frontdoor_subregion_logits"),
            target,
        ),
        "frontdoor_balanced_logits": _balanced_subregion_mediator_loss(
            outputs.get("frontdoor_logits"),
            target,
        ),
        "frontdoor_router": _frontdoor_router_advantage_loss(
            outputs.get("causal_mediator_router_logits"),
            outputs.get("frontdoor_base_logits"),
            outputs.get("frontdoor_logits"),
            target,
        ),
        "region_causal_logits": _subregion_prior_loss(
            outputs.get("region_causal_logits"),
            target,
            args,
        ),
        "region_causal_balanced": _balanced_subregion_mediator_loss(
            outputs.get("region_causal_logits"),
            target,
        ),
        "nested_causal_region": _spatial_region_head_loss(
            outputs.get("nested_causal_region_logits"),
            target,
            args,
        ),
        "nested_causal_subregion": _subregion_prior_loss(
            outputs.get("nested_causal_subregion_logits"),
            target,
            args,
        ),
        "nested_causal_balanced_region": _balanced_region_mediator_loss(
            outputs.get("nested_causal_region_logits"),
            target,
        ),
        "nested_causal_balanced_subregion": _balanced_subregion_mediator_loss(
            outputs.get("nested_causal_subregion_logits"),
            target,
        ),
        "nested_causal_router": _nested_causal_router_advantage_loss(
            outputs.get("nested_causal_router_logits"),
            outputs.get("nested_causal_base_logits"),
            outputs.get("nested_causal_subregion_logits"),
            target,
        ),
        "cascade_error": _cascade_error_correction_loss(
            outputs.get("cascade_logits"),
            outputs.get("cascade_base_logits"),
            target,
            args,
        ),
        "boundary_mediator": _boundary_mediator_loss(
            outputs.get("boundary_logits"),
            target,
        ),
        "causal_refiner_sparsity": _causal_refiner_sparsity_loss(
            outputs.get("causal_refiner_delta"),
            outputs.get("disease_attention_logits"),
        ),
        "cite_contrastive": _cite_contrastive_loss(
            outputs.get("cite_anchor"),
            outputs.get("cite_positive"),
            outputs.get("cite_negative"),
            outputs.get("cite_bank_negative"),
            temperature=getattr(args, "cite_temperature", 0.2),
        ),
    }
    proxy_terms.update(_sdd_treatment_terms(outputs, batch))
    proxy_terms.update(_sdd_outcome_terms(outputs, target, scale=getattr(args, "region_volume_scale", 1000.0)))
    proxy_terms["sdd_imbalance"] = _sdd_imbalance_term(outputs, batch)
    for name, value in proxy_terms.items():
        if value is not None:
            terms[name] = value
    return terms


def _weighted_total(terms: dict[str, Tensor], args: argparse.Namespace) -> Tensor:
    weights = {
        "seg": args.lambda_seg,
        "region": args.lambda_region_loss,
        "adjustment": args.lambda_adjustment,
        "adjustment_region": args.lambda_region_loss,
        "context_swap": args.lambda_context_swap,
        "context_swap_region": args.lambda_context_swap_region,
        "context_swap_consistency": args.lambda_context_swap_consistency,
        "teacher_distill": args.lambda_teacher_distill,
        "adjusted_teacher_distill": args.lambda_adjusted_teacher_distill,
        "context_swap_teacher_distill": args.lambda_context_swap_teacher_distill,
        "context_stability": args.lambda_context_stability,
        "context_response": args.lambda_context_response,
        "context_proxy": args.lambda_context_proxy,
        "disease_proxy": args.lambda_disease_proxy,
        "annotation_proxy": args.lambda_annotation_proxy,
        "context_from_disease_adversary": getattr(args, "lambda_context_from_disease_adversary", 0.0),
        "disease_from_context_adversary": getattr(args, "lambda_disease_from_context_adversary", 0.0),
        "region_volume_proxy": getattr(args, "lambda_region_volume_proxy", 0.0),
        "region_from_context_adversary": getattr(args, "lambda_region_from_context_adversary", 0.0),
        "sdd_context_teacher": getattr(args, "lambda_sdd_context_teacher", 0.0),
        "sdd_region_teacher": getattr(args, "lambda_sdd_region_teacher", 0.0),
        "sdd_context_distill": getattr(args, "lambda_sdd_context_distill", 0.0),
        "sdd_region_distill": getattr(args, "lambda_sdd_region_distill", 0.0),
        "sdd_treatment": getattr(args, "lambda_sdd_treatment", 0.0),
        "sdd_treatment_disentangle": getattr(args, "lambda_sdd_treatment_disentangle", 0.0),
        "sdd_outcome": getattr(args, "lambda_sdd_outcome", 0.0),
        "sdd_outcome_disentangle": getattr(args, "lambda_sdd_outcome_disentangle", 0.0),
        "sdd_imbalance": getattr(args, "lambda_sdd_imbalance", 0.0),
        "cite_contrastive": getattr(args, "lambda_cite_contrastive", 0.0),
        "spatial_disease_attention": getattr(args, "lambda_spatial_disease_attention", 0.0),
        "spatial_region_head": getattr(args, "lambda_spatial_region_head", 0.0),
        "subregion_prior": getattr(args, "lambda_subregion_prior", 0.0),
        "prototype_mediator": getattr(args, "lambda_prototype_mediator", 0.0),
        "category_confounder": getattr(args, "lambda_category_confounder", 0.0),
        "modality_prior": getattr(args, "lambda_modality_prior", 0.0),
        "frontdoor_region": getattr(args, "lambda_frontdoor_region", 0.0),
        "frontdoor_subregion": getattr(args, "lambda_frontdoor_subregion", 0.0),
        "frontdoor_logits": getattr(args, "lambda_frontdoor_logits", 0.0),
        "frontdoor_balanced_region": getattr(args, "lambda_frontdoor_balanced_mediator", 0.0),
        "frontdoor_balanced_subregion": getattr(args, "lambda_frontdoor_balanced_mediator", 0.0),
        "frontdoor_balanced_logits": getattr(args, "lambda_frontdoor_balanced_mediator", 0.0),
        "frontdoor_router": getattr(args, "lambda_frontdoor_router", 0.0),
        "region_causal_logits": getattr(args, "lambda_region_causal_logits", 0.0),
        "region_causal_balanced": getattr(args, "lambda_region_causal_balanced", 0.0),
        "region_causal_teacher_distill": getattr(args, "lambda_region_causal_teacher_distill", 0.0),
        "nested_causal_region": getattr(args, "lambda_nested_causal_region", 0.0),
        "nested_causal_subregion": getattr(args, "lambda_nested_causal_subregion", 0.0),
        "nested_causal_balanced_region": getattr(args, "lambda_nested_causal_balanced", 0.0),
        "nested_causal_balanced_subregion": getattr(args, "lambda_nested_causal_balanced", 0.0),
        "nested_causal_router": getattr(args, "lambda_nested_causal_router", 0.0),
        "cascade_error": getattr(args, "lambda_cascade_error", 0.0),
        "boundary_mediator": getattr(args, "lambda_boundary_mediator", 0.0),
        "causal_refiner_sparsity": getattr(args, "lambda_causal_refiner_sparsity", 0.0),
        "style_intervention_seg": getattr(args, "lambda_style_intervention_seg", 0.0),
        "style_intervention_seg_region": getattr(args, "lambda_style_intervention_seg", 0.0),
        "style_intervention_consistency": getattr(args, "lambda_style_intervention_consistency", 0.0),
        "style_disease_invariance": getattr(args, "lambda_style_disease_invariance", 0.0),
        "style_context_response": getattr(args, "lambda_style_context_response", 0.0),
        "feature_intervention_seg": getattr(args, "lambda_feature_intervention_seg", 0.0),
        "feature_intervention_seg_region": getattr(args, "lambda_feature_intervention_seg", 0.0),
        "feature_intervention_consistency": getattr(args, "lambda_feature_intervention_consistency", 0.0),
        "lesion_paste_seg": getattr(args, "lambda_lesion_paste_seg", 0.0),
        "lesion_paste_seg_region": getattr(args, "lambda_lesion_paste_seg", 0.0),
        "lesion_paste_effect": getattr(args, "lambda_lesion_intervention_effect", 0.0),
        "lesion_paste_spatial_attention": getattr(args, "lambda_lesion_mediator", 0.0),
        "lesion_paste_spatial_region": getattr(args, "lambda_lesion_mediator", 0.0),
        "lesion_paste_prototype_mediator": getattr(args, "lambda_lesion_mediator", 0.0),
        "lesion_paste_boundary_mediator": getattr(args, "lambda_lesion_mediator", 0.0),
        "lesion_erase_seg": getattr(args, "lambda_lesion_erase_seg", 0.0),
        "lesion_erase_seg_region": getattr(args, "lambda_lesion_erase_seg", 0.0),
        "lesion_erase_effect": getattr(args, "lambda_lesion_intervention_effect", 0.0),
        "lesion_erase_spatial_attention": getattr(args, "lambda_lesion_mediator", 0.0),
        "lesion_erase_spatial_region": getattr(args, "lambda_lesion_mediator", 0.0),
        "lesion_erase_prototype_mediator": getattr(args, "lambda_lesion_mediator", 0.0),
        "lesion_erase_boundary_mediator": getattr(args, "lambda_lesion_mediator", 0.0),
        "orthogonal": args.lambda_orthogonal,
    }
    total = torch.zeros((), device=next(iter(terms.values())).device)
    for name, value in terms.items():
        total = total + float(weights.get(name, 1.0)) * value
    return total


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _float_terms(terms: dict[str, Tensor]) -> dict[str, float]:
    return {f"loss/{name}": float(value.detach().cpu()) for name, value in terms.items()}


def _run_train_epoch(
    model: CausalSegFormer3D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    context_bank: Tensor | None,
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
    teacher_model: nn.Module | None,
) -> dict[str, float]:
    model.train()
    loss_logs: list[dict[str, float]] = []
    metric_items: list[dict[str, float]] = []
    bank_device = context_bank.to(device) if context_bank is not None else None
    for batch_idx, batch in enumerate(tqdm(loader, desc="causal-train", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
        _add_context_swap_outputs(model, outputs, bank_device, args)
        teacher_logits = _teacher_logits(teacher_model, image)
        if teacher_logits is not None:
            outputs["teacher_logits"] = teacher_logits
        terms = _causal_loss_terms(outputs, batch, args, proxy_layout)
        total = _weighted_total(terms, args)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        item = {"loss/total": float(total.detach().cpu())}
        item.update(_float_terms(terms))
        loss_logs.append(item)
        metrics = brats_region_metrics(outputs["logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold)
        if "adjusted_logits" in outputs:
            metrics.update(
                _prefix_metrics(
                    brats_region_metrics(outputs["adjusted_logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold),
                    "adjusted",
                )
            )
        if "context_swap_logits" in outputs:
            metrics.update(
                _prefix_metrics(
                    brats_region_metrics(outputs["context_swap_logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold),
                    "context_swap",
                )
            )
        metric_items.append(metrics)
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
    return {**_average_metric_dicts(loss_logs), **_average_metric_dicts(metric_items)}


@torch.no_grad()
def _run_eval_epoch(
    model: CausalSegFormer3D,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    context_bank: Tensor | None,
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
    teacher_model: nn.Module | None,
) -> dict[str, float]:
    model.eval()
    loss_logs: list[dict[str, float]] = []
    metric_items: list[dict[str, float]] = []
    bank_device = context_bank.to(device) if context_bank is not None else None
    for batch_idx, batch in enumerate(tqdm(loader, desc="causal-val", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
        _add_context_swap_outputs(model, outputs, bank_device, args)
        teacher_logits = _teacher_logits(teacher_model, image)
        if teacher_logits is not None:
            outputs["teacher_logits"] = teacher_logits
        terms = _causal_loss_terms(outputs, batch, args, proxy_layout)
        total = _weighted_total(terms, args)

        item = {"loss/total": float(total.detach().cpu())}
        item.update(_float_terms(terms))
        loss_logs.append(item)
        metrics = brats_region_metrics(outputs["logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold)
        if "adjusted_logits" in outputs:
            metrics.update(
                _prefix_metrics(
                    brats_region_metrics(outputs["adjusted_logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold),
                    "adjusted",
                )
            )
        if "context_swap_logits" in outputs:
            metrics.update(
                _prefix_metrics(
                    brats_region_metrics(outputs["context_swap_logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold),
                    "context_swap",
                )
            )
        metric_items.append(metrics)
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break
    return {**_average_metric_dicts(loss_logs), **_average_metric_dicts(metric_items)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train causal SegFormer3D on UTSW with Pearl-style proxy adjustment.")
    parser.add_argument("--baseline-checkpoint", default="runs/segformer3d_utsw_base/best.pt")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--data-root", default="data/brats/PKG - UTSW-Glioma/UTSW-Glioma")
    parser.add_argument("--metadata-path")
    parser.add_argument("--splits-json")
    parser.add_argument("--output-dir", default="runs/segformer3d_utsw_causal")
    parser.add_argument("--model-size", choices=["tiny", "base"], default="base")
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--causal-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--prefer-manual-seg", action="store_true")
    parser.add_argument("--use-ants-modalities", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument("--context-swap-strategy", choices=["none", "random", "nearest", "farthest"], default="none")
    parser.add_argument("--context-bank-refresh-epochs", type=int, default=1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-stability-margin", type=float, default=0.03)
    parser.add_argument("--context-response-target", type=float, default=0.0)
    parser.add_argument("--region-volume-scale", type=float, default=1000.0)
    parser.add_argument("--channel-loss-weights", default="1.0,1.0,1.0")
    parser.add_argument("--region-loss-weights", default="1.0,1.5,2.5")
    parser.add_argument("--distill-channel-weights", default="1.0,1.0,2.0")
    parser.add_argument("--proxy-loss-mode", choices=["mse", "typed"], default="typed")
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-region-loss", type=float, default=0.0)
    parser.add_argument("--lambda-adjustment", type=float, default=0.25)
    parser.add_argument("--lambda-context-swap", type=float, default=0.0)
    parser.add_argument("--lambda-context-swap-region", type=float, default=0.0)
    parser.add_argument("--lambda-context-swap-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-teacher-distill", type=float, default=0.0)
    parser.add_argument("--lambda-adjusted-teacher-distill", type=float, default=0.0)
    parser.add_argument("--lambda-context-swap-teacher-distill", type=float, default=0.0)
    parser.add_argument("--lambda-context-stability", type=float, default=0.02)
    parser.add_argument("--lambda-context-response", type=float, default=0.0)
    parser.add_argument("--lambda-context-proxy", type=float, default=0.05)
    parser.add_argument("--lambda-disease-proxy", type=float, default=0.05)
    parser.add_argument("--lambda-annotation-proxy", type=float, default=0.01)
    parser.add_argument("--lambda-frontdoor-router", type=float, default=0.0)
    parser.add_argument("--lambda-orthogonal", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")
    splits = _load_or_make_splits(args, baseline_checkpoint, data_root)
    _save_json(splits, output_dir / "splits.json")
    _save_json(vars(args), output_dir / "config.json")
    _save_json(asdict(default_utsw_scm()), output_dir / "scm.json")

    train_loader = _make_loader(data_root, splits["train"], args, shuffle=True)
    val_loader = _make_loader(data_root, splits["val"], args, shuffle=False)
    bank_loader = _make_loader(data_root, splits["train"], args, shuffle=False)
    _require_metadata(train_loader.dataset, args.allow_missing_metadata)
    proxy_layout = _metadata_layout(train_loader.dataset)

    device = _resolve_device(args.device)
    model = _build_model_from_dataset(args, train_loader.dataset)
    _load_baseline_backbone(model, Path(args.baseline_checkpoint))
    model.to(device)
    teacher_model = _build_teacher_model(args.teacher_checkpoint, train_loader.dataset, device)
    optimizer = _build_optimizer(model, args)

    best_adjusted_dice = float("-inf")
    context_bank: Tensor | None = None
    for epoch in range(1, args.epochs + 1):
        _set_backbone_trainable(model, epoch > args.freeze_backbone_epochs)
        if context_bank is None or (args.context_bank_refresh_epochs > 0 and (epoch - 1) % args.context_bank_refresh_epochs == 0):
            context_bank = build_context_bank(
                model,
                bank_loader,
                device,
                max_contexts=args.context_bank_size,
                max_batches=args.max_context_bank_batches,
                sampling=args.context_bank_sampling,
                seed=args.seed + epoch,
            )
        train_metrics = _run_train_epoch(model, train_loader, optimizer, device, args, context_bank, proxy_layout, teacher_model)
        val_metrics = _run_eval_epoch(model, val_loader, device, args, context_bank, proxy_layout, teacher_model)
        monitor = float(val_metrics.get("adjusted/brats/mean_dice", val_metrics.get("brats/mean_dice", float("-inf"))))
        log = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        _save_json(log, output_dir / f"epoch_{epoch:03d}.json")
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss/total"),
                "val_loss": val_metrics.get("loss/total"),
                "val_factual_mean_dice": val_metrics.get("brats/mean_dice"),
                "val_adjusted_mean_dice": val_metrics.get("adjusted/brats/mean_dice"),
            }
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "splits": splits,
            "proxy_dims": _metadata_dims(train_loader.dataset),
            "proxy_layout": proxy_layout,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if monitor > best_adjusted_dice:
            best_adjusted_dice = monitor
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_adjusted_brats_mean_dice": best_adjusted_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
