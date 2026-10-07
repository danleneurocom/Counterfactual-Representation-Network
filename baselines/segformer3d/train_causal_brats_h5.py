from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.segformer3d.causal import CausalSegFormer3D, build_causal_segformer3d, default_utsw_scm
from baselines.segformer3d.evaluate_causal_brats_h5 import BraTSH5VolumeDataset
from baselines.segformer3d.train_brats_h5 import _make_dataset, _volume_ids
from baselines.segformer3d.train_causal_utsw import (
    _add_context_swap_outputs,
    _build_optimizer,
    _causal_loss_terms,
    _float_terms,
    _prefix_metrics,
    _set_backbone_trainable,
    _weighted_total,
    build_context_bank,
)
from baselines.segformer3d.train_utsw import _average_metric_dicts, _resolve_device, _save_json
from crn.metrics import brats_region_metrics


def _make_loader(dataset: BraTSH5VolumeDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def _build_model(args: argparse.Namespace) -> CausalSegFormer3D:
    return build_causal_segformer3d(
        model_size=args.model_size,
        latent_dim=args.latent_dim,
        num_classes=3,
        context_proxy_dim=0,
        disease_proxy_dim=0,
        annotation_proxy_dim=0,
    )


def _load_baseline_backbone(model: CausalSegFormer3D, checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if any(str(key).startswith("backbone.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("backbone."): value
            for key, value in state_dict.items()
            if str(key).startswith("backbone.")
        }
    model.load_baseline_state_dict(state_dict, strict_backbone=True)
    return checkpoint


def _mean(items: list[float]) -> float:
    return float(sum(items) / len(items)) if items else float("nan")


def _run_train_epoch(
    model: CausalSegFormer3D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    context_bank: Tensor | None,
) -> dict[str, float]:
    model.train()
    loss_logs: list[dict[str, float]] = []
    metric_items: list[dict[str, float]] = []
    bank_device = context_bank.to(device) if context_bank is not None else None
    for batch_idx, batch in enumerate(tqdm(loader, desc="brats-causal-train", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
        _add_context_swap_outputs(model, outputs, bank_device, args)
        terms = _causal_loss_terms(outputs, batch, args, proxy_layout=None)
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
) -> dict[str, float]:
    model.eval()
    loss_logs: list[dict[str, float]] = []
    metric_items: list[dict[str, float]] = []
    bank_device = context_bank.to(device) if context_bank is not None else None
    for batch_idx, batch in enumerate(tqdm(loader, desc="brats-causal-val", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
        _add_context_swap_outputs(model, outputs, bank_device, args)
        terms = _causal_loss_terms(outputs, batch, args, proxy_layout=None)
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
    parser = argparse.ArgumentParser(description="Train Causal SegFormer3D on BraTS2020 HDF5 volumes.")
    parser.add_argument("--baseline-checkpoint", default="runs/segformer3d_brats_h5_base/best.pt")
    parser.add_argument("--train-csv", default="data/brats/brats_train.csv")
    parser.add_argument("--val-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    parser.add_argument("--output-dir", default="runs/segformer3d_brats_h5_causal")
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
    parser.add_argument("--limit-train-volumes", type=int)
    parser.add_argument("--limit-val-volumes", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--slice-col", default="slice")
    parser.add_argument("--h5-image-key", default="image")
    parser.add_argument("--h5-mask-key", default="mask")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument("--context-swap-strategy", choices=["none", "random", "nearest", "farthest"], default="none")
    parser.add_argument("--context-bank-refresh-epochs", type=int, default=1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-stability-margin", type=float, default=0.03)
    parser.add_argument("--context-response-target", type=float, default=0.0)
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
    parser.add_argument("--lambda-context-proxy", type=float, default=0.0)
    parser.add_argument("--lambda-disease-proxy", type=float, default=0.0)
    parser.add_argument("--lambda-annotation-proxy", type=float, default=0.0)
    parser.add_argument("--lambda-orthogonal", type=float, default=0.01)
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
    _save_json(asdict(default_utsw_scm()), output_dir / "scm.json")

    train_dataset = _make_dataset(args.train_csv, train_ids, args)
    val_dataset = _make_dataset(args.val_csv, val_ids, args)
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    val_loader = _make_loader(val_dataset, args, shuffle=False)
    bank_loader = _make_loader(train_dataset, args, shuffle=False)

    device = _resolve_device(args.device)
    model = _build_model(args)
    _load_baseline_backbone(model, args.baseline_checkpoint)
    model.to(device)
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
        train_metrics = _run_train_epoch(model, train_loader, optimizer, device, args, context_bank)
        val_metrics = _run_eval_epoch(model, val_loader, device, args, context_bank)
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
            "proxy_dims": {"context_proxy_dim": 0, "disease_proxy_dim": 0, "annotation_proxy_dim": 0},
            "proxy_layout": None,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if monitor > best_adjusted_dice:
            best_adjusted_dice = monitor
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_adjusted_brats_mean_dice": best_adjusted_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
