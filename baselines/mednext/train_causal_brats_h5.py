from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from baselines.mednext.causal import CausalMedNeXt, build_causal_mednext
from baselines.mednext.train_causal_utsw import (
    _add_causal_args,
    _run_eval_epoch,
    _run_train_epoch,
    build_category_confounder_dictionary,
    build_sdd_cite_bank,
)
from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.train_brats_h5 import _make_dataset, _make_loader, _volume_ids
from baselines.segformer3d.train_causal_utsw import _build_optimizer, _set_backbone_trainable
from baselines.segformer3d.train_utsw import _resolve_device, _save_json


def _build_model(args: argparse.Namespace) -> CausalMedNeXt:
    return build_causal_mednext(
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        latent_dim=args.latent_dim,
        num_classes=3,
        base_channels=args.base_channels,
        modulation_scale=args.modulation_scale,
        causal_residual_scale=args.causal_residual_scale,
        contrastive_dim=args.contrastive_dim,
        spatial_refiner_scale=args.spatial_refiner_scale,
        region_fusion_scale=args.region_fusion_scale,
        prototype_dim=args.prototype_dim,
        prototype_fusion_scale=args.prototype_fusion_scale,
        prototype_temperature=args.prototype_temperature,
        category_confounder_scale=args.category_confounder_scale,
        category_confounder_temperature=args.category_confounder_temperature,
        modality_prior_scale=args.modality_prior_scale,
        logit_calibration_scale=args.logit_calibration_scale,
        cascade_refiner_scale=args.cascade_refiner_scale,
        frontdoor_mediator_scale=args.frontdoor_mediator_scale,
        frontdoor_residual_scale=args.frontdoor_residual_scale,
        region_causal_bottleneck_scale=args.region_causal_bottleneck_scale,
        region_causal_background_leak=args.region_causal_background_leak,
        region_causal_base=args.region_causal_base,
        region_causal_mask_source=args.region_causal_mask_source,
        context_proxy_dim=0,
        disease_proxy_dim=0,
        annotation_proxy_dim=0,
    )


def _load_baseline_backbone(model: CausalMedNeXt, checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_baseline_state_dict(state_dict, strict_backbone=True)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train causal MedNeXt on BraTS2020 HDF5 volumes.")
    parser.add_argument("--baseline-checkpoint", default="runs/mednext_brats_h5_s_k3/best.pt")
    parser.add_argument("--train-csv", default="data/brats/brats_train.csv")
    parser.add_argument("--val-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    parser.add_argument("--output-dir", default="runs/mednext_brats_h5_causal_s_k3")
    parser.add_argument("--model-id", choices=["S", "B", "M", "L"], default="S")
    parser.add_argument("--kernel-size", type=int, choices=[3, 5], default=3)
    parser.add_argument("--base-channels", type=int)
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
    _add_causal_args(parser)
    parser.set_defaults(lambda_context_proxy=0.0, lambda_disease_proxy=0.0, lambda_annotation_proxy=0.0)
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
    train_loader: DataLoader = _make_loader(train_dataset, args, shuffle=True)
    val_loader: DataLoader = _make_loader(val_dataset, args, shuffle=False)
    bank_loader: DataLoader = _make_loader(train_dataset, args, shuffle=False)

    device = _resolve_device(args.device)
    model = _build_model(args)
    _load_baseline_backbone(model, args.baseline_checkpoint)
    model.to(device)
    optimizer = _build_optimizer(model, args)

    best_adjusted_dice = float("-inf")
    contrastive_bank: dict[str, Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        _set_backbone_trainable(model, epoch > args.freeze_backbone_epochs)
        category_report = build_category_confounder_dictionary(
            model,
            bank_loader,
            device,
            max_batches=args.max_category_confounder_batches
            if args.max_category_confounder_batches is not None
            else args.max_context_bank_batches,
            threshold=args.threshold,
        )
        if contrastive_bank is None or (args.context_bank_refresh_epochs > 0 and (epoch - 1) % args.context_bank_refresh_epochs == 0):
            contrastive_bank = build_sdd_cite_bank(
                model,
                bank_loader,
                device,
                max_contexts=args.context_bank_size,
                max_batches=args.max_context_bank_batches,
                sampling=args.context_bank_sampling,
                seed=args.seed + epoch,
            )
        context_bank = None if contrastive_bank is None else contrastive_bank["z_c"]
        train_metrics = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            context_bank,
            contrastive_bank,
            proxy_layout=None,
        )
        val_metrics = _run_eval_epoch(
            model,
            val_loader,
            device,
            args,
            context_bank,
            contrastive_bank,
            proxy_layout=None,
        )
        monitor = float(val_metrics.get("adjusted/brats/mean_dice", val_metrics.get("brats/mean_dice", float("-inf"))))
        _save_json(
            {"epoch": epoch, "category_confounders": category_report, "train": train_metrics, "val": val_metrics},
            output_dir / f"epoch_{epoch:03d}.json",
        )
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
            "category_confounders": category_report,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if monitor > best_adjusted_dice:
            best_adjusted_dice = monitor
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_adjusted_brats_mean_dice": best_adjusted_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
