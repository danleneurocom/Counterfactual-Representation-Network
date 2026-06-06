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
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_causal_utsw import (
    _metadata_dims,
    _metadata_layout,
    _proxy_loss,
    _require_metadata,
    build_context_bank,
)
from baselines.segformer3d.train_utsw import (
    _average_metric_dicts,
    _case_ids,
    _make_splits,
    _resolve_device,
    _save_json,
)
from cpa_seg3d import CPASeg3D, boundary_targets_from_subregions, build_cpa_seg3d, region_targets_from_subregions
from crn.metrics import brats_region_metrics


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_loader(root: Path, case_ids: list[str], args: argparse.Namespace, shuffle: bool) -> DataLoader:
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


def _load_or_make_splits(args: argparse.Namespace, data_root: Path) -> dict[str, list[str]]:
    if args.splits_json:
        return _load_json(Path(args.splits_json))
    return _make_splits(_case_ids(data_root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)


def _build_model_from_dataset(args: argparse.Namespace, dataset: UTSWGliomaDataset) -> CPASeg3D:
    return build_cpa_seg3d(
        model_size=args.model_size,
        latent_dim=args.latent_dim,
        num_classes=3,
        decoder_variant=args.decoder_variant,
        **_metadata_dims(dataset),
    )


def _load_segformer_checkpoint(model: CPASeg3D, checkpoint_path: str | None) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    return model.load_segformer3d_state_dict(state_dict)


def _set_encoder_trainable(model: CPASeg3D, trainable: bool) -> None:
    for parameter in model.backbone.segformer_encoder.parameters():
        parameter.requires_grad = trainable


def _build_optimizer(model: CPASeg3D, args: argparse.Namespace) -> AdamW:
    encoder_params: list[Tensor] = []
    method_params: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.segformer_encoder."):
            encoder_params.append(parameter)
        else:
            method_params.append(parameter)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr, "name": "segformer3d_encoder"})
    if method_params:
        groups.append({"params": method_params, "lr": args.lr, "name": "cpa_seg3d_method"})
    return AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def _dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _bce_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + _dice_loss(logits, target)


def _boundary_loss(logits: Tensor, target: Tensor) -> Tensor:
    prevalence = target.mean().detach()
    pos_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-4)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return bce + _dice_loss(logits, target)


def _orthogonality_loss(z_d: Tensor, z_c: Tensor) -> Tensor:
    z_d_norm = F.normalize(z_d, dim=1)
    z_c_norm = F.normalize(z_c, dim=1)
    return (z_d_norm * z_c_norm).sum(dim=1).pow(2).mean()


def _bounded_context_shift_loss(factual_logits: Tensor, adjusted_logits: Tensor, margin: float) -> Tensor:
    shift = (torch.sigmoid(factual_logits) - torch.sigmoid(adjusted_logits)).abs().mean()
    return torch.relu(shift - float(margin))


def _as_tensor(outputs: dict[str, Any], key: str) -> Tensor:
    value = outputs[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected output {key!r} to be a Tensor.")
    return value


def _deep_supervision_loss(outputs: dict[str, Any], target: Tensor) -> Tensor | None:
    deep_logits = outputs.get("deep_logits")
    if not isinstance(deep_logits, list) or not deep_logits:
        return None
    losses = [_bce_dice_loss(logits, target) for logits in deep_logits if isinstance(logits, Tensor)]
    if not losses:
        return None
    weights = torch.linspace(0.5, 1.0, steps=len(losses), device=target.device, dtype=target.dtype)
    stacked = torch.stack(losses)
    return (stacked * weights).sum() / weights.sum()


def _loss_terms(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    args: argparse.Namespace,
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Tensor]:
    target = batch["mask"].to(_as_tensor(outputs, "logits").device)
    region_target = region_targets_from_subregions(target)
    boundary_target = boundary_targets_from_subregions(target, kernel_size=args.boundary_kernel_size)
    terms: dict[str, Tensor] = {
        "seg": _bce_dice_loss(_as_tensor(outputs, "logits"), target),
        "region": _bce_dice_loss(_as_tensor(outputs, "region_logits"), region_target),
        "boundary": _boundary_loss(_as_tensor(outputs, "boundary_logits"), boundary_target),
        "orthogonal": _orthogonality_loss(_as_tensor(outputs, "z_d"), _as_tensor(outputs, "z_c")),
    }
    deep = _deep_supervision_loss(outputs, target)
    if deep is not None:
        terms["deep"] = deep
    if "adjusted_logits" in outputs:
        terms["adjustment"] = _bce_dice_loss(_as_tensor(outputs, "adjusted_logits"), target)
        terms["context_stability"] = _bounded_context_shift_loss(
            _as_tensor(outputs, "logits"),
            _as_tensor(outputs, "adjusted_logits"),
            margin=args.context_stability_margin,
        )
    if "adjusted_region_logits" in outputs:
        terms["adjusted_region"] = _bce_dice_loss(_as_tensor(outputs, "adjusted_region_logits"), region_target)

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
    }
    for name, value in proxy_terms.items():
        if value is not None:
            terms[name] = value
    return terms


def _weighted_total(terms: dict[str, Tensor], args: argparse.Namespace) -> Tensor:
    weights = {
        "seg": args.lambda_seg,
        "region": args.lambda_region,
        "boundary": args.lambda_boundary,
        "deep": args.lambda_deep,
        "adjustment": args.lambda_adjustment,
        "adjusted_region": args.lambda_adjusted_region,
        "context_stability": args.lambda_context_stability,
        "context_proxy": args.lambda_context_proxy,
        "disease_proxy": args.lambda_disease_proxy,
        "annotation_proxy": args.lambda_annotation_proxy,
        "orthogonal": args.lambda_orthogonal,
    }
    total = torch.zeros((), device=next(iter(terms.values())).device)
    for name, value in terms.items():
        total = total + float(weights.get(name, 1.0)) * value
    return total


def _float_terms(terms: dict[str, Tensor]) -> dict[str, float]:
    return {f"loss/{name}": float(value.detach().cpu()) for name, value in terms.items()}


def _run_epoch(
    model: CPASeg3D,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    context_bank: Tensor | None,
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    bank_device = context_bank.to(device) if context_bank is not None else None
    loss_items: list[dict[str, float]] = []
    metric_items: list[dict[str, float]] = []
    max_batches = args.max_train_batches if is_train else args.max_val_batches
    desc = "cpa-train" if is_train else "cpa-val"
    for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        with torch.set_grad_enabled(is_train):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
            terms = _loss_terms(outputs, batch, args, proxy_layout)
            total = _weighted_total(terms, args)
            if optimizer is not None:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        item = {"loss/total": float(total.detach().cpu())}
        item.update(_float_terms(terms))
        loss_items.append(item)
        metric_items.append(brats_region_metrics(_as_tensor(outputs, "logits").detach().cpu(), target.detach().cpu(), threshold=args.threshold))
        if max_batches is not None and batch_idx >= max_batches:
            break
    return {**_average_metric_dicts(loss_items), **_average_metric_dicts(metric_items)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPA-Seg3D on UTSW glioma volumes.")
    parser.add_argument("--data-root", default="data/brats/PKG - UTSW-Glioma/UTSW-Glioma")
    parser.add_argument("--metadata-path")
    parser.add_argument("--splits-json")
    parser.add_argument("--segformer-checkpoint")
    parser.add_argument("--output-dir", default="runs/cpa_seg3d_utsw")
    parser.add_argument("--model-size", choices=["tiny", "base"], default="base")
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--decoder-variant", choices=["lite", "unet"], default="lite")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
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
    parser.add_argument("--context-bank-refresh-epochs", type=int, default=1)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-stability-margin", type=float, default=0.03)
    parser.add_argument("--boundary-kernel-size", type=int, default=3)
    parser.add_argument("--proxy-loss-mode", choices=["mse", "typed"], default="typed")
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-region", type=float, default=0.3)
    parser.add_argument("--lambda-boundary", type=float, default=0.1)
    parser.add_argument("--lambda-deep", type=float, default=0.2)
    parser.add_argument("--lambda-adjustment", type=float, default=0.25)
    parser.add_argument("--lambda-adjusted-region", type=float, default=0.1)
    parser.add_argument("--lambda-context-stability", type=float, default=0.02)
    parser.add_argument("--lambda-context-proxy", type=float, default=0.05)
    parser.add_argument("--lambda-disease-proxy", type=float, default=0.05)
    parser.add_argument("--lambda-annotation-proxy", type=float, default=0.01)
    parser.add_argument("--lambda-orthogonal", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_or_make_splits(args, data_root)
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
    loading_report = _load_segformer_checkpoint(model, args.segformer_checkpoint)
    model.to(device)
    optimizer = _build_optimizer(model, args)

    best_dice = float("-inf")
    context_bank: Tensor | None = None
    for epoch in range(1, args.epochs + 1):
        _set_encoder_trainable(model, epoch > args.freeze_encoder_epochs)
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
        train_metrics = _run_epoch(model, train_loader, device, args, context_bank, proxy_layout, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, device, args, context_bank, proxy_layout, optimizer=None)
        monitor = float(val_metrics.get("brats/mean_dice", float("-inf")))
        log = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        _save_json(log, output_dir / f"epoch_{epoch:03d}.json")
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss/total"),
                "val_loss": val_metrics.get("loss/total"),
                "val_brats_mean_dice": val_metrics.get("brats/mean_dice"),
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
            "segformer_loading_report": loading_report,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if monitor > best_dice:
            best_dice = monitor
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_brats_mean_dice": best_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
