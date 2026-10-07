from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.segformer3d.evaluate_causal_brats_h5 import BraTSH5VolumeDataset
from baselines.segformer3d.train_utsw import (
    _average_metric_dicts,
    _build_model,
    _mean,
    _resolve_device,
    _save_json,
    _segmentation_loss,
)
from crn.metrics import brats_region_metrics


def _volume_ids(csv_path: str | Path, volume_col: str = "volume", limit: int | None = None) -> list[int]:
    frame = pd.read_csv(csv_path, usecols=[volume_col])
    ids = sorted(int(value) for value in frame[volume_col].drop_duplicates().tolist())
    if limit is not None:
        ids = ids[: int(limit)]
    if not ids:
        raise ValueError(f"No BraTS volumes found in {csv_path}")
    return ids


def _make_dataset(csv_path: str | Path, volume_ids: list[int], args: argparse.Namespace) -> BraTSH5VolumeDataset:
    return BraTSH5VolumeDataset(
        csv_path=csv_path,
        data_root=args.data_root,
        volume_size=args.volume_size,
        volume_ids=volume_ids,
        path_col=args.path_col,
        volume_col=args.volume_col,
        slice_col=args.slice_col,
        image_key=args.h5_image_key,
        mask_key=args.h5_mask_key,
        crop_margin=args.crop_margin,
    )


def _make_loader(dataset: BraTSH5VolumeDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def _load_initial_weights(model: nn.Module, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if any(str(key).startswith("backbone.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("backbone."): value
            for key, value in state_dict.items()
            if str(key).startswith("backbone.")
        }
    model.load_state_dict(state_dict, strict=True)


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="brats-train", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = _segmentation_loss(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(logits.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
    metrics = _average_metric_dicts(metric_items)
    metrics["loss"] = _mean(losses)
    return metrics


@torch.no_grad()
def _run_eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="brats-val", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        logits = model(image)
        loss = _segmentation_loss(logits, target)
        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(logits.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break
    metrics = _average_metric_dicts(metric_items)
    metrics["loss"] = _mean(losses)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer3D on BraTS2020 HDF5 volumes.")
    parser.add_argument("--train-csv", default="data/brats/brats_train.csv")
    parser.add_argument("--val-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    parser.add_argument("--output-dir", default="runs/segformer3d_brats_h5_base")
    parser.add_argument("--init-checkpoint", help="Optional SegFormer3D checkpoint for fine-tuning instead of from-scratch training.")
    parser.add_argument("--model-size", choices=["tiny", "base"], default="base")
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-train-volumes", type=int)
    parser.add_argument("--limit-val-volumes", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--slice-col", default="slice")
    parser.add_argument("--h5-image-key", default="image")
    parser.add_argument("--h5-mask-key", default="mask")
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids = _volume_ids(args.train_csv, args.volume_col, args.limit_train_volumes)
    val_ids = _volume_ids(args.val_csv, args.volume_col, args.limit_val_volumes)
    splits = {"train": train_ids, "val": val_ids}
    _save_json(splits, output_dir / "splits.json")
    _save_json(vars(args), output_dir / "config.json")

    train_dataset = _make_dataset(args.train_csv, train_ids, args)
    val_dataset = _make_dataset(args.val_csv, val_ids, args)
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    val_loader = _make_loader(val_dataset, args, shuffle=False)

    device = _resolve_device(args.device)
    model = _build_model(args.model_size).to(device)
    _load_initial_weights(model, args.init_checkpoint)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_train_epoch(model, train_loader, optimizer, device, args)
        val_metrics = _run_eval_epoch(model, val_loader, device, args)
        monitor = float(val_metrics.get("brats/mean_dice", float("-inf")))
        log = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        _save_json(log, output_dir / f"epoch_{epoch:03d}.json")
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss"),
                "val_loss": val_metrics.get("loss"),
                "val_brats_mean_dice": val_metrics.get("brats/mean_dice"),
            }
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "splits": splits,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if monitor > best_dice:
            best_dice = monitor
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_brats_mean_dice": best_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
