from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from crn.data import make_dataloader
from crn.losses import compute_crn_losses
from crn.metrics import (
    binary_segmentation_metrics,
    brats_region_metrics,
    classification_metrics,
    multilabel_segmentation_metrics,
    summarize_metrics,
)
from crn.train import build_model
from crn.utils import load_yaml, move_batch_to_device, resolve_device, save_json


def _load_config(checkpoint: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    if config_path:
        return load_yaml(config_path)
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain a saved config; pass --config explicitly.")
    return checkpoint["config"]


def _segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, data_config: dict[str, Any], threshold: float) -> dict[str, float]:
    mask_mode = data_config.get("mask_mode")
    if target.ndim == 3:
        return binary_segmentation_metrics(logits, target, threshold)
    if target.ndim == 4 and target.shape[1] == 1:
        return binary_segmentation_metrics(logits, target, threshold)
    if mask_mode == "brats_subregions":
        return brats_region_metrics(
            logits,
            target,
            threshold=threshold,
            channel_names=data_config.get("brats_channel_names"),
            region_channels=data_config.get("brats_region_channels"),
        )
    return multilabel_segmentation_metrics(logits, target, threshold, data_config.get("segmentation_channel_names"))


def evaluate(
    checkpoint_path: str | Path,
    split: str = "val",
    config_path: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_batches: int | None = None,
    device_name: str = "auto",
    threshold: float = 0.5,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = _load_config(checkpoint, config_path)

    data_config = dict(config["data"])
    train_config = dict(config.get("training", {}))
    data_config["batch_size"] = batch_size or train_config.get("batch_size", 8)
    data_config["num_workers"] = num_workers if num_workers is not None else train_config.get("num_workers", 0)

    device = resolve_device(device_name if device_name != "auto" else train_config.get("device", "auto"))
    loader = make_dataloader(data_config, split, shuffle=False)
    iterator = islice(loader, max_batches) if max_batches is not None else loader
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cls_logits: list[torch.Tensor] = []
    cls_targets: list[torch.Tensor] = []
    seg_logits: list[torch.Tensor] = []
    seg_targets: list[torch.Tensor] = []
    loss_logs: list[dict[str, float]] = []

    with torch.no_grad():
        for batch in tqdm(iterator, total=total_batches, leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"])
            _, logs = compute_crn_losses(model, batch, outputs, config.get("loss", {}))
            loss_logs.append({key: float(value.detach().cpu()) for key, value in logs.items()})

            if "logits" in outputs and "label" in batch:
                cls_logits.append(outputs["logits"].detach().cpu())
                cls_targets.append(batch["label"].detach().cpu())
            if "seg_logits" in outputs and "mask" in batch:
                seg_logits.append(outputs["seg_logits"].detach().cpu())
                seg_targets.append(batch["mask"].detach().cpu())

    metrics: dict[str, Any] = summarize_metrics(loss_logs)
    if cls_logits:
        metrics.update(classification_metrics(torch.cat(cls_logits, dim=0), torch.cat(cls_targets, dim=0), threshold))
    if seg_logits:
        metrics.update(_segmentation_metrics(torch.cat(seg_logits, dim=0), torch.cat(seg_targets, dim=0), data_config, threshold))

    metrics["eval/epoch"] = int(checkpoint.get("epoch", -1))
    metrics["eval/split"] = split
    metrics["eval/checkpoint"] = str(Path(checkpoint_path))

    if output_path is None:
        checkpoint_path = Path(checkpoint_path)
        output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_{split}_metrics.json")
    save_json(metrics, output_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CRN checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint, e.g. runs/brats_crn/best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to evaluate.")
    parser.add_argument("--config", help="Optional config path. Uses checkpoint config by default.")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, help="Optional num_workers override.")
    parser.add_argument("--max-batches", type=int, help="Evaluate only this many batches.")
    parser.add_argument("--device", default="auto", help="Device override: auto, cpu, cuda, mps.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary metrics.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        split=args.split,
        config_path=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        device_name=args.device,
        threshold=args.threshold,
        output_path=args.output,
    )
    print(metrics)


if __name__ == "__main__":
    main()
