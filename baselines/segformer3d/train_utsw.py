from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.segformer3d import SegFormer3D
from baselines.segformer3d.data import UTSWGliomaDataset
from crn.metrics import brats_region_metrics


def _resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(model_size: str, num_classes: int = 3) -> SegFormer3D:
    if model_size == "tiny":
        return SegFormer3D(
            in_channels=4,
            sr_ratios=[4, 2, 1, 1],
            embed_dims=[4, 8, 16, 32],
            patch_kernel_size=[7, 3, 3, 3],
            patch_stride=[4, 2, 2, 2],
            patch_padding=[3, 1, 1, 1],
            mlp_ratios=[2, 2, 2, 2],
            num_heads=[1, 1, 2, 4],
            depths=[1, 1, 1, 1],
            decoder_head_embedding_dim=8,
            num_classes=num_classes,
            decoder_dropout=0.0,
        )
    if model_size == "base":
        return SegFormer3D(in_channels=4, num_classes=num_classes)
    raise ValueError(f"Unknown model_size: {model_size}")


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + _dice_loss(logits, target)


def _case_ids(root: Path, limit: int | None) -> list[str]:
    ids = sorted(path.name for path in root.iterdir() if path.is_dir())
    if limit is not None:
        ids = ids[: int(limit)]
    if len(ids) < 3:
        raise ValueError("Need at least 3 cases to create train/val/test splits.")
    return ids


def _make_splits(case_ids: list[str], seed: int, val_fraction: float, test_fraction: float) -> dict[str, list[str]]:
    rng = random.Random(seed)
    ids = list(case_ids)
    rng.shuffle(ids)
    n_total = len(ids)
    n_test = max(1, round(n_total * test_fraction))
    n_val = max(1, round(n_total * val_fraction))
    if n_test + n_val >= n_total:
        n_test = 1
        n_val = 1
    test = sorted(ids[:n_test])
    val = sorted(ids[n_test : n_test + n_val])
    train = sorted(ids[n_test + n_val :])
    if not train:
        raise ValueError("Split produced no training cases; reduce val/test fractions or increase case limit.")
    return {"train": train, "val": val, "test": test}


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


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
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def _mean(items: list[float]) -> float:
    return float(sum(items) / len(items)) if items else float("nan")


def _average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(set().union(*(item.keys() for item in items)))
    return {
        key: _mean([float(item[key]) for item in items if key in item])
        for key in keys
    }


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = _segmentation_loss(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(logits.detach().cpu(), target.detach().cpu(), threshold=0.5))
        if max_batches is not None and batch_idx >= max_batches:
            break
    metrics = _average_metric_dicts(metric_items)
    metrics["loss"] = _mean(losses)
    return metrics


@torch.no_grad()
def _run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        logits = model(image)
        loss = _segmentation_loss(logits, target)
        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(logits.detach().cpu(), target.detach().cpu(), threshold=0.5))
        if max_batches is not None and batch_idx >= max_batches:
            break
    metrics = _average_metric_dicts(metric_items)
    metrics["loss"] = _mean(losses)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer3D from scratch on UTSW glioma volumes.")
    parser.add_argument("--data-root", default="data/brats/PKG - UTSW-Glioma/UTSW-Glioma")
    parser.add_argument("--output-dir", default="runs/segformer3d_utsw")
    parser.add_argument("--model-size", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--volume-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--limit-cases", type=int, help="Limit case count for quick smoke runs.")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--prefer-manual-seg", action="store_true")
    parser.add_argument("--use-ants-modalities", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _make_splits(_case_ids(data_root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)
    _save_json(splits, output_dir / "splits.json")
    _save_json(vars(args), output_dir / "config.json")

    train_loader = _make_loader(data_root, splits["train"], args, shuffle=True)
    val_loader = _make_loader(data_root, splits["val"], args, shuffle=False)

    device = _resolve_device(args.device)
    model = _build_model(args.model_size).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_train_epoch(model, train_loader, optimizer, device, args.max_train_batches)
        val_metrics = _run_eval_epoch(model, val_loader, device, args.max_val_batches)
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
