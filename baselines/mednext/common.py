from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mednext import build_mednext_segmenter
from baselines.segformer3d.train_utsw import _average_metric_dicts, _dice_loss, _mean, _resolve_device
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


def _parse_loss_weights(value: str | None, count: int) -> list[float]:
    if value is None:
        return [1.0] + [0.5 ** idx for idx in range(1, count)]
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) < count:
        weights.extend([weights[-1]] * (count - len(weights)))
    return weights[:count]


def segmentation_loss(output: Tensor | list[Tensor] | tuple[Tensor, ...], target: Tensor, ds_weights: str | None = None) -> Tensor:
    outputs = [output] if isinstance(output, Tensor) else list(output)
    weights = _parse_loss_weights(ds_weights, len(outputs))
    losses: list[Tensor] = []
    for logits, weight in zip(outputs, weights, strict=True):
        target_i = target
        if tuple(logits.shape[-3:]) != tuple(target.shape[-3:]):
            target_i = F.interpolate(target, size=logits.shape[-3:], mode="nearest")
        loss_i = F.binary_cross_entropy_with_logits(logits, target_i) + _dice_loss(logits, target_i)
        losses.append(loss_i * float(weight))
    return torch.stack(losses).sum() / max(sum(weights), 1e-6)


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
        loss = segmentation_loss(output, target, args.deep_supervision_weights)
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
    for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        output = model(image)
        loss = segmentation_loss(output, target, args.deep_supervision_weights)
        logits = main_logits(output).detach().cpu()
        target_cpu = target.detach().cpu()
        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(logits, target_cpu, threshold=args.threshold))
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break
    metrics = _average_metric_dicts(metric_items)
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
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--pin-memory", action="store_true")


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
    "load_model_for_eval",
    "main_logits",
    "run_eval_epoch",
    "run_train_epoch",
    "save_json",
    "segmentation_loss",
    "_resolve_device",
]
