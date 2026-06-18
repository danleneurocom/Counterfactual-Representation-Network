from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mednext import build_mednext_segmenter
from baselines.mednext.calibration import (
    CALIBRATION_OBJECTIVES,
    BratsRegionThresholdSweep,
    brats_region_probabilities,
    brats_region_targets,
    parse_threshold_candidates,
    prefix_metrics,
)
from baselines.segformer3d.train_utsw import _average_metric_dicts, _mean, _resolve_device
from crn.metrics import brats_region_metrics


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main_logits(output: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if not output:
        raise ValueError("MedNeXt returned an empty output list.")
    return output[0]


def parse_mirror_tta_axes(spec: str | None) -> tuple[int, ...]:
    if spec is None:
        return ()
    text = str(spec).strip().lower()
    if not text or text in {"0", "none", "off", "false"}:
        return ()
    aliases = {
        "d": 2,
        "z": 2,
        "h": 3,
        "y": 3,
        "w": 4,
        "x": 4,
        "2": 2,
        "3": 3,
        "4": 4,
    }
    raw_parts = text.replace(";", ",").replace("|", ",").split(",")
    if len(raw_parts) == 1 and all(char in aliases for char in text):
        raw_parts = list(text)
    axes: list[int] = []
    for raw_part in raw_parts:
        part = raw_part.strip()
        if not part:
            continue
        if part not in aliases:
            raise ValueError("--mirror-tta-axes must use d,h,w/z,y,x or tensor dims 2,3,4.")
        axis = aliases[part]
        if axis not in axes:
            axes.append(axis)
    return tuple(sorted(axes))


def _mirror_axis_subsets(axes: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [subset for length in range(1, len(axes) + 1) for subset in combinations(axes, length)]


@torch.no_grad()
def mirror_tta_logits(
    forward_logits: Any,
    image: Tensor,
    axes: tuple[int, ...],
    *,
    base_logits: Tensor | None = None,
) -> Tensor:
    if not axes:
        return base_logits if base_logits is not None else forward_logits(image)
    total = base_logits if base_logits is not None else forward_logits(image)
    count = 1
    for subset in _mirror_axis_subsets(axes):
        logits = forward_logits(image.flip(subset)).flip(subset)
        total = total + logits
        count += 1
    return total / float(count)


def _parse_loss_weights(value: str | None, count: int) -> list[float]:
    if value is None:
        return [1.0] + [0.5 ** idx for idx in range(1, count)]
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) < count:
        weights.extend([weights[-1]] * (count - len(weights)))
    return weights[:count]


def _parse_channel_weights(value: str | None, count: int, name: str) -> tuple[float, ...]:
    if value is None:
        return tuple(1.0 for _ in range(count))
    weights = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if len(weights) != count:
        raise ValueError(f"{name} must contain {count} comma-separated floats, got {value!r}")
    return weights


def _channel_weight_tensor(weights: tuple[float, ...], logits: Tensor) -> Tensor:
    shape = (1, len(weights)) + (1,) * (logits.ndim - 2)
    return torch.as_tensor(weights, device=logits.device, dtype=logits.dtype).view(shape)


def _channel_weighted_bce_loss(logits: Tensor, target: Tensor, weights: tuple[float, ...]) -> Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * _channel_weight_tensor(weights, logits)).mean()


def _voxel_balanced_bce_loss(logits: Tensor, target: Tensor, weights: tuple[float, ...], max_pos_weight: float) -> Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    dims = tuple(idx for idx in range(target.ndim) if idx != 1)
    positives = target.sum(dim=dims).clamp_min(1.0)
    negatives = (1.0 - target).sum(dim=dims).clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(1.0, float(max_pos_weight))
    view_shape = (1, target.shape[1]) + (1,) * (target.ndim - 2)
    balance = torch.where(target > 0.5, pos_weight.view(view_shape), torch.ones_like(target))
    channel = _channel_weight_tensor(weights, logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    combined = balance * channel
    return (loss * combined).sum() / combined.sum().clamp_min(1.0)


def _channel_weighted_dice_loss(logits: Tensor, target: Tensor, weights: tuple[float, ...], eps: float = 1e-6) -> Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    channel_weights = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    weighted = (1.0 - dice) * channel_weights.view(1, -1)
    return weighted.sum(dim=1).div(channel_weights.sum().clamp_min(eps)).mean()


def _focal_tversky_loss(
    logits: Tensor,
    target: Tensor,
    weights: tuple[float, ...],
    alpha: float,
    beta: float,
    gamma: float,
    eps: float = 1e-6,
) -> Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, logits.ndim))
    true_pos = (probs * target).sum(dim=dims)
    false_neg = ((1.0 - probs) * target).sum(dim=dims)
    false_pos = (probs * (1.0 - target)).sum(dim=dims)
    tversky = (true_pos + eps) / (true_pos + float(alpha) * false_neg + float(beta) * false_pos + eps)
    channel_weights = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    loss = (1.0 - tversky).clamp_min(eps).pow(float(gamma)) * channel_weights.view(1, -1)
    return loss.sum(dim=1).div(channel_weights.sum().clamp_min(eps)).mean()


def _region_weighted_dice_loss(logits: Tensor, target: Tensor, weights: tuple[float, ...], eps: float = 1e-6) -> Tensor:
    region_probs = brats_region_probabilities(logits)
    region_target = brats_region_targets(target).to(device=logits.device, dtype=logits.dtype)
    dims = tuple(range(2, region_probs.ndim))
    region_weights = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    intersection = (region_probs * region_target).sum(dim=dims)
    denominator = region_probs.sum(dim=dims) + region_target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    weighted = (1.0 - dice) * region_weights.view(1, -1)
    return weighted.sum(dim=1).div(region_weights.sum().clamp_min(eps)).mean()


def _region_volume_prior_loss(logits: Tensor, target: Tensor, scale: float = 1000.0) -> Tensor:
    region_probs = brats_region_probabilities(logits).mean(dim=(2, 3, 4))
    region_target = brats_region_targets(target).to(device=logits.device, dtype=logits.dtype).mean(dim=(2, 3, 4))
    scale = max(float(scale), 1e-6)
    return F.smooth_l1_loss(torch.log1p(scale * region_probs), torch.log1p(scale * region_target))


def registered_modality_consistency_metrics(
    native_logits: Tensor,
    registered_logits: Tensor,
    *,
    fused_logits: Tensor | None = None,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    native_prob = torch.sigmoid(native_logits.detach().float())
    registered_prob = torch.sigmoid(registered_logits.detach().float())
    prob_delta = (native_prob - registered_prob).abs()
    region_native = brats_region_probabilities(native_logits.detach().float())
    region_registered = brats_region_probabilities(registered_logits.detach().float())
    region_delta = (region_native - region_registered).abs()
    native_pred = (region_native >= float(threshold)).float()
    registered_pred = (region_registered >= float(threshold)).float()
    dims = tuple(range(2, native_pred.ndim))
    intersection = (native_pred * registered_pred).sum(dim=dims)
    denominator = native_pred.sum(dim=dims) + registered_pred.sum(dim=dims)
    agreement = (2.0 * intersection + eps) / (denominator + eps)
    names = ("WT", "TC", "ET")
    metrics: dict[str, float] = {
        "registered_consistency/prob_l1": float(prob_delta.mean().item()),
        "registered_consistency/prob_mse": float((native_prob - registered_prob).square().mean().item()),
        "registered_consistency/region_prob_l1": float(region_delta.mean().item()),
        "registered_consistency/region_prob_mse": float((region_native - region_registered).square().mean().item()),
    }
    metrics["registered_consistency/prob_similarity"] = float(1.0 - metrics["registered_consistency/prob_l1"])
    metrics["registered_consistency/region_prob_similarity"] = float(1.0 - metrics["registered_consistency/region_prob_l1"])
    agreement_values: list[float] = []
    for idx, name in enumerate(names):
        value = float(agreement[:, idx].mean().item())
        metrics[f"registered_consistency/brats/{name}/agreement_dice"] = value
        agreement_values.append(value)
    mean_agreement = float(sum(agreement_values) / len(agreement_values))
    metrics["registered_consistency/brats/mean_agreement_dice"] = mean_agreement
    metrics["registered_consistency/stability_score"] = float(
        0.5 * (mean_agreement + metrics["registered_consistency/region_prob_similarity"])
    )
    if fused_logits is not None:
        fused_prob = torch.sigmoid(fused_logits.detach().float())
        metrics["registered_consistency/fused_native_prob_l1"] = float((fused_prob - native_prob).abs().mean().item())
        metrics["registered_consistency/fused_registered_prob_l1"] = float((fused_prob - registered_prob).abs().mean().item())
    return metrics


def _single_segmentation_loss(
    logits: Tensor,
    target: Tensor,
    args: argparse.Namespace | None,
    *,
    include_region: bool = True,
) -> Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    channel_weights = _parse_channel_weights(getattr(args, "channel_loss_weights", None), int(target.shape[1]), "--channel-loss-weights")
    mode = str(getattr(args, "seg_loss_mode", "bce_dice"))
    if mode == "balanced_focal":
        loss = _voxel_balanced_bce_loss(
            logits,
            target,
            channel_weights,
            max_pos_weight=float(getattr(args, "balanced_bce_max_pos_weight", 50.0)),
        ) + _focal_tversky_loss(
            logits,
            target,
            channel_weights,
            alpha=float(getattr(args, "focal_tversky_alpha", 0.7)),
            beta=float(getattr(args, "focal_tversky_beta", 0.3)),
            gamma=float(getattr(args, "focal_tversky_gamma", 0.75)),
        )
    elif mode == "bce_dice":
        loss = _channel_weighted_bce_loss(logits, target, channel_weights) + _channel_weighted_dice_loss(logits, target, channel_weights)
    else:
        raise ValueError(f"Unknown MedNeXt segmentation loss mode {mode!r}")

    if include_region and float(getattr(args, "lambda_region_loss", 0.0)) > 0.0:
        region_weights = _parse_channel_weights(getattr(args, "region_loss_weights", "1.0,1.5,2.5"), 3, "--region-loss-weights")
        loss = loss + float(getattr(args, "lambda_region_loss", 0.0)) * _region_weighted_dice_loss(logits, target, region_weights)

    volume_weight = float(getattr(args, "lambda_volume_prior_loss", 0.0))
    if volume_weight > 0.0:
        loss = loss + volume_weight * _region_volume_prior_loss(
            logits,
            target,
            scale=float(getattr(args, "volume_prior_scale", 1000.0)),
        )
    return loss


def segmentation_terms(logits: Tensor, target: Tensor, args: argparse.Namespace, prefix: str) -> dict[str, Tensor]:
    terms = {prefix: _single_segmentation_loss(logits, target, args, include_region=False)}
    if float(getattr(args, "lambda_region_loss", 0.0)) > 0.0:
        name = "region" if prefix == "seg" else f"{prefix}_region"
        region_weights = _parse_channel_weights(getattr(args, "region_loss_weights", "1.0,1.5,2.5"), 3, "--region-loss-weights")
        terms[name] = _region_weighted_dice_loss(logits, target, region_weights)
    return terms


def segmentation_loss(
    output: Tensor | list[Tensor] | tuple[Tensor, ...],
    target: Tensor,
    ds_weights: str | None = None,
    args: argparse.Namespace | None = None,
) -> Tensor:
    outputs = [output] if isinstance(output, Tensor) else list(output)
    weights = _parse_loss_weights(ds_weights, len(outputs))
    losses: list[Tensor] = []
    for logits, weight in zip(outputs, weights, strict=True):
        target_i = target
        if tuple(logits.shape[-3:]) != tuple(target.shape[-3:]):
            target_i = F.interpolate(target, size=logits.shape[-3:], mode="nearest")
        loss_i = _single_segmentation_loss(logits, target_i, args)
        losses.append(loss_i * float(weight))
    return torch.stack(losses).sum() / max(sum(weights), 1e-6)


def _set_output_bias(module: nn.Module, bias: Tensor) -> list[str]:
    updated: list[str] = []
    for name in ("out0", "out1", "out2", "out3", "out4"):
        head = getattr(module, name, None)
        if isinstance(head, nn.Conv3d) and head.bias is not None and head.bias.numel() == bias.numel():
            head.bias.copy_(bias.to(device=head.bias.device, dtype=head.bias.dtype))
            updated.append(name)
    backbone = getattr(module, "backbone", None)
    if isinstance(backbone, nn.Module):
        updated.extend([f"backbone.{name}" for name in _set_output_bias(backbone, bias)])
    return updated


@torch.no_grad()
def initialize_output_bias_from_loader(model: nn.Module, loader: DataLoader, args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "init_output_bias_from_data", False)):
        return {}
    max_batches = getattr(args, "output_bias_init_batches", 16)
    max_batches = None if max_batches is None else max(1, int(max_batches))
    sums: Tensor | None = None
    voxel_count = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="mednext-bias-init", leave=False), start=1):
        target = batch["mask"].detach().float()
        if sums is None:
            sums = torch.zeros(target.shape[1], dtype=torch.float64)
        sums += target.sum(dim=(0, 2, 3, 4)).to(dtype=torch.float64)
        voxel_count += int(target.shape[0] * target.shape[2] * target.shape[3] * target.shape[4])
        if max_batches is not None and batch_idx >= max_batches:
            break
    if sums is None or voxel_count <= 0:
        return {"updated_heads": [], "channel_prevalence": [], "channel_bias": []}
    min_prob = float(getattr(args, "output_bias_min_prob", 1e-4))
    max_prob = float(getattr(args, "output_bias_max_prob", 0.5))
    prevalence = (sums / float(voxel_count)).clamp(min=min_prob, max=max_prob).to(dtype=torch.float32)
    prior_strength = max(0.0, float(getattr(args, "output_bias_prior_strength", 1.0)))
    bias = torch.logit(prevalence) * prior_strength
    updated_heads = _set_output_bias(model, bias)
    return {
        "updated_heads": updated_heads,
        "channel_prevalence": [float(value) for value in prevalence.tolist()],
        "channel_bias": [float(value) for value in bias.tolist()],
        "num_voxels": float(voxel_count),
        "prior_strength": prior_strength,
    }


def build_model_from_args(args: argparse.Namespace) -> nn.Module:
    return build_mednext_segmenter(
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        in_channels=4,
        num_classes=3,
        deep_supervision=args.deep_supervision,
        base_channels=args.base_channels,
    )


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    config["architecture"] = "MedNeXtSegmenter"
    return config


def initialize_model_from_checkpoint(model: nn.Module, checkpoint_path: str | Path | None) -> dict[str, Any]:
    if not checkpoint_path:
        return {}
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint": str(path),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "loaded_keys": len(state_dict),
    }


def checkpoint_monitor(metrics: dict[str, float], args: argparse.Namespace, *, prefer_adjusted: bool = False) -> tuple[float, str]:
    keys: list[str] = []
    prefer_registered_tta = bool(getattr(args, "checkpoint_registered_modality_tta", False))
    registered_selector = str(getattr(args, "checkpoint_registered_modality_selector", "dice"))
    if prefer_registered_tta and registered_selector == "agreement":
        keys.append("registered_consistency/brats/mean_agreement_dice")
    elif prefer_registered_tta and registered_selector == "region-prob-similarity":
        keys.append("registered_consistency/region_prob_similarity")
    elif prefer_registered_tta and registered_selector == "stability":
        keys.append("registered_consistency/stability_score")
    elif prefer_registered_tta and registered_selector == "prob-response":
        keys.append("registered_consistency/prob_l1")
    elif prefer_registered_tta and registered_selector == "region-prob-response":
        keys.append("registered_consistency/region_prob_l1")
    elif prefer_registered_tta and registered_selector == "val-loss":
        keys.append("selection/negative_loss")
    elif registered_selector != "dice":
        raise ValueError(f"Unsupported registered modality checkpoint selector: {registered_selector}")
    if parse_threshold_candidates(getattr(args, "checkpoint_calibration_thresholds", None)):
        calibration_objective = str(getattr(args, "checkpoint_calibration_objective", "mean"))
        calibration_metric = (
            "threshold/objective_score" if calibration_objective != "mean" else "brats/mean_dice"
        )
        if prefer_registered_tta:
            keys.append(f"registered_tta_sweep_region_calibrated/{calibration_metric}")
        if prefer_adjusted:
            keys.append(f"adjusted_sweep_region_calibrated/{calibration_metric}")
        keys.append(f"sweep_region_calibrated/{calibration_metric}")
        if calibration_metric != "brats/mean_dice":
            if prefer_registered_tta:
                keys.append("registered_tta_sweep_region_calibrated/brats/mean_dice")
            if prefer_adjusted:
                keys.append("adjusted_sweep_region_calibrated/brats/mean_dice")
            keys.append("sweep_region_calibrated/brats/mean_dice")
    if prefer_registered_tta:
        keys.append("registered_tta/brats/mean_dice")
    if prefer_adjusted:
        keys.append("adjusted/brats/mean_dice")
    keys.append("brats/mean_dice")
    for key in keys:
        if key in metrics:
            return float(metrics[key]), key
    return float("-inf"), keys[-1]


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    desc: str = "mednext-train",
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        loss = segmentation_loss(output, target, args.deep_supervision_weights, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        logits = main_logits(output).detach().cpu()
        target_cpu = target.detach().cpu()
        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(logits, target_cpu, threshold=args.threshold))
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
    metrics = _average_metric_dicts(metric_items)
    metrics["loss"] = _mean(losses)
    return metrics


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    desc: str = "mednext-val",
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    calibration_candidates = parse_threshold_candidates(getattr(args, "checkpoint_calibration_thresholds", None))
    calibration_sweep = (
        BratsRegionThresholdSweep(
            calibration_candidates,
            objective=getattr(args, "checkpoint_calibration_objective", "mean"),
        )
        if calibration_candidates
        else None
    )
    for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        output = model(image)
        loss = segmentation_loss(output, target, args.deep_supervision_weights, args)
        logits = main_logits(output).detach().cpu()
        target_cpu = target.detach().cpu()
        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(logits, target_cpu, threshold=args.threshold))
        if calibration_sweep is not None:
            calibration_sweep.update(logits, target_cpu)
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break
    metrics = _average_metric_dicts(metric_items)
    if calibration_sweep is not None:
        metrics.update(prefix_metrics(calibration_sweep.summary(), "sweep_region_calibrated"))
    metrics["loss"] = _mean(losses)
    return metrics


def add_common_train_args(parser: argparse.ArgumentParser, default_output_dir: str, default_volume_size: int) -> None:
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--model-id", choices=["S", "B", "M", "L"], default="S")
    parser.add_argument("--kernel-size", type=int, choices=[3, 5], default=3)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--volume-size", type=int, default=default_volume_size)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--deep-supervision-weights", default="1.0,0.5,0.25,0.125,0.0625")
    parser.add_argument("--seg-loss-mode", choices=["bce_dice", "balanced_focal"], default="bce_dice")
    parser.add_argument("--channel-loss-weights", default="1.0,1.0,1.0")
    parser.add_argument("--region-loss-weights", default="1.0,1.5,2.5")
    parser.add_argument("--lambda-region-loss", type=float, default=0.0)
    parser.add_argument("--balanced-bce-max-pos-weight", type=float, default=50.0)
    parser.add_argument("--focal-tversky-alpha", type=float, default=0.7)
    parser.add_argument("--focal-tversky-beta", type=float, default=0.3)
    parser.add_argument("--focal-tversky-gamma", type=float, default=0.75)
    parser.add_argument("--lambda-volume-prior-loss", type=float, default=0.0)
    parser.add_argument("--volume-prior-scale", type=float, default=1000.0)
    parser.add_argument("--init-output-bias-from-data", action="store_true")
    parser.add_argument("--output-bias-init-batches", type=int, default=16)
    parser.add_argument("--output-bias-min-prob", type=float, default=1e-4)
    parser.add_argument("--output-bias-max-prob", type=float, default=0.5)
    parser.add_argument("--output-bias-prior-strength", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--disk-cache-dir", help="Optional MedNeXt-only disk cache for preprocessed dataset items.")
    parser.add_argument("--init-checkpoint", help="Optional MedNeXt baseline checkpoint to initialize model weights before training.")
    parser.add_argument("--checkpoint-calibration-thresholds", help="Optional WT/TC/ET threshold grid for calibrated validation checkpoint selection.")
    parser.add_argument(
        "--checkpoint-calibration-objective",
        choices=CALIBRATION_OBJECTIVES,
        default="mean",
        help="Objective used when choosing validation thresholds from --checkpoint-calibration-thresholds.",
    )


def load_model_for_eval(checkpoint: dict[str, Any], args: argparse.Namespace) -> nn.Module:
    config = dict(checkpoint.get("config", {}))
    model = build_mednext_segmenter(
        model_id=str(args.model_id or config.get("model_id", "S")),
        kernel_size=int(args.kernel_size or config.get("kernel_size", 3)),
        in_channels=4,
        num_classes=3,
        deep_supervision=bool(config.get("deep_supervision", False)),
        base_channels=args.base_channels if args.base_channels is not None else config.get("base_channels"),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


__all__ = [
    "add_common_train_args",
    "build_model_from_args",
    "checkpoint_config",
    "checkpoint_monitor",
    "initialize_output_bias_from_loader",
    "initialize_model_from_checkpoint",
    "load_model_for_eval",
    "main_logits",
    "mirror_tta_logits",
    "parse_mirror_tta_axes",
    "registered_modality_consistency_metrics",
    "run_eval_epoch",
    "run_train_epoch",
    "save_json",
    "segmentation_loss",
    "segmentation_terms",
    "_resolve_device",
]
