from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mednext.causal import build_causal_mednext
from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.evaluate_causal_brats_h5 import BraTSH5VolumeDataset, _context_overlap, _save_json, _segmentation_loss, _volume_metrics
from baselines.segformer3d.train_causal_utsw import build_context_bank, _prefix_metrics
from baselines.segformer3d.train_utsw import _average_metric_dicts, _mean, _resolve_device
from crn.metrics import brats_region_metrics


def _make_loader(dataset: BraTSH5VolumeDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _build_model(checkpoint: dict[str, Any], args: argparse.Namespace):
    config = dict(checkpoint.get("config", {}))
    model = build_causal_mednext(
        model_id=str(args.model_id or config.get("model_id", "S")),
        kernel_size=int(args.kernel_size or config.get("kernel_size", 3)),
        latent_dim=int(args.latent_dim or config.get("latent_dim", 128)),
        num_classes=3,
        base_channels=args.base_channels if args.base_channels is not None else config.get("base_channels"),
        modulation_scale=float(config.get("modulation_scale", 0.1)),
        causal_residual_scale=float(config.get("causal_residual_scale", 0.2)),
        contrastive_dim=int(config.get("contrastive_dim", 64)),
        spatial_refiner_scale=float(config.get("spatial_refiner_scale", 0.5)),
        region_fusion_scale=float(config.get("region_fusion_scale", 0.0)),
        prototype_dim=int(config.get("prototype_dim", 32)),
        prototype_fusion_scale=float(config.get("prototype_fusion_scale", 0.0)),
        prototype_temperature=float(config.get("prototype_temperature", 0.1)),
        category_confounder_scale=float(config.get("category_confounder_scale", 0.0)),
        category_confounder_temperature=float(config.get("category_confounder_temperature", 0.2)),
        modality_prior_scale=float(config.get("modality_prior_scale", 0.0)),
        logit_calibration_scale=float(config.get("logit_calibration_scale", 0.0)),
        cascade_refiner_scale=float(config.get("cascade_refiner_scale", 0.0)),
        frontdoor_mediator_scale=float(config.get("frontdoor_mediator_scale", 0.0)),
        frontdoor_residual_scale=float(config.get("frontdoor_residual_scale", 0.25)),
        region_causal_bottleneck_scale=float(config.get("region_causal_bottleneck_scale", 0.0)),
        region_causal_background_leak=float(config.get("region_causal_background_leak", 0.05)),
        region_causal_base=str(config.get("region_causal_base", "prior")),
        region_causal_mask_source=str(config.get("region_causal_mask_source", "spatial")),
        context_proxy_dim=0,
        disease_proxy_dim=0,
        annotation_proxy_dim=0,
    )
    model.load_compatible_state_dict(checkpoint["model"])
    return model


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    volume_size = int(args.volume_size or config.get("volume_size", 128))
    eval_dataset = BraTSH5VolumeDataset(
        csv_path=args.brats_csv,
        data_root=args.data_root,
        volume_size=volume_size,
        limit_volumes=args.max_volumes,
        path_col=args.path_col,
        volume_col=args.volume_col,
        slice_col=args.slice_col,
        image_key=args.h5_image_key,
        mask_key=args.h5_mask_key,
        crop_margin=args.crop_margin,
    )
    bank_dataset = BraTSH5VolumeDataset(
        csv_path=args.context_csv or args.brats_csv,
        data_root=args.data_root,
        volume_size=volume_size,
        limit_volumes=args.max_context_volumes,
        path_col=args.path_col,
        volume_col=args.volume_col,
        slice_col=args.slice_col,
        image_key=args.h5_image_key,
        mask_key=args.h5_mask_key,
        crop_margin=args.crop_margin,
    )

    device = _resolve_device(args.device)
    model = _build_model(checkpoint, args).to(device)
    model.eval()
    eval_loader = _make_loader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    bank_loader = _make_loader(bank_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    context_bank = build_context_bank(
        model,
        bank_loader,
        device,
        max_contexts=args.context_bank_size,
        max_batches=args.max_context_bank_batches,
        sampling=args.context_bank_sampling,
        seed=args.seed,
    )
    bank_device = context_bank.to(device) if context_bank is not None else None

    factual_losses: list[float] = []
    adjusted_losses: list[float] = []
    factual_batch_metrics: list[dict[str, float]] = []
    adjusted_batch_metrics: list[dict[str, float]] = []
    region_causal_batch_metrics: list[dict[str, float]] = []
    factual_volume_metrics: list[dict[str, float]] = []
    adjusted_volume_metrics: list[dict[str, float]] = []
    context_shifts: list[float] = []
    nearest_context_distances: list[float] = []

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"mednext-brats-causal-eval:{args.split_name}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
        logits = outputs["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("CausalMedNeXt output 'logits' must be a tensor.")
        factual_losses.append(float(_segmentation_loss(logits, target).detach().cpu()))
        factual_batch_metrics.append(brats_region_metrics(logits.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
        factual_volume_metrics.extend(_volume_metrics(logits, target, args.threshold))
        region_causal_logits = outputs.get("region_causal_logits")
        if isinstance(region_causal_logits, Tensor):
            region_causal_batch_metrics.append(
                brats_region_metrics(region_causal_logits.detach().cpu(), target.detach().cpu(), threshold=args.threshold)
            )

        z_c = outputs.get("z_c")
        if isinstance(z_c, Tensor):
            nearest_context_distances.extend(_context_overlap(z_c, bank_device).get("overlap/nearest_context_l2", []))

        adjusted = outputs.get("adjusted_logits")
        if isinstance(adjusted, Tensor):
            adjusted_losses.append(float(_segmentation_loss(adjusted, target).detach().cpu()))
            adjusted_batch_metrics.append(brats_region_metrics(adjusted.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
            adjusted_volume_metrics.extend(_volume_metrics(adjusted, target, args.threshold))
            context_shifts.append(float((torch.sigmoid(logits) - torch.sigmoid(adjusted)).abs().mean().detach().cpu()))

        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    scm = default_utsw_scm()
    metrics: dict[str, Any] = {
        "method": "Causal MedNeXt on BraTS H5",
        "causal_question": scm.question.query,
        "causal_estimand": scm.question.estimand,
        "causal_warning": "BraTS H5 evaluation has no metadata proxy columns; proxy losses are intentionally omitted.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split_name,
        "brats_csv": str(args.brats_csv),
        "context_csv": str(args.context_csv or args.brats_csv),
        "threshold": float(args.threshold),
        "num_cases": float(len(factual_volume_metrics)),
        "context_bank_size": float(0 if context_bank is None else context_bank.shape[0]),
        "factual/loss": _mean(factual_losses),
    }
    metrics.update(_average_metric_dicts(factual_batch_metrics))
    if region_causal_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(region_causal_batch_metrics), "region_causal"))
    metrics.update({f"volume/{key}": value for key, value in _average_metric_dicts(factual_volume_metrics).items()})
    if adjusted_losses:
        metrics["adjusted/loss"] = _mean(adjusted_losses)
        metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_batch_metrics), "adjusted"))
        metrics.update({f"adjusted/volume/{key}": value for key, value in _average_metric_dicts(adjusted_volume_metrics).items()})
        metrics["intervention/context_adjustment_mean_abs_prob_shift"] = _mean(context_shifts)
        metrics["intervention/adjusted_minus_factual_mean_dice"] = float(metrics.get("adjusted/brats/mean_dice", float("nan"))) - float(metrics.get("brats/mean_dice", float("nan")))
    if nearest_context_distances:
        distances = torch.tensor(nearest_context_distances, dtype=torch.float32)
        metrics["overlap/nearest_context_l2_mean"] = float(distances.mean())
        metrics["overlap/nearest_context_l2_max"] = float(distances.max())
        metrics["overlap/nearest_context_l2_p90"] = float(torch.quantile(distances, 0.9))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained causal MedNeXt checkpoint on BraTS2020 HDF5 volumes.")
    parser.add_argument("--checkpoint", default="runs/mednext_brats_h5_causal_s_k3/best.pt")
    parser.add_argument("--brats-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--context-csv", default="data/brats/brats_train.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    parser.add_argument("--output-json")
    parser.add_argument("--split-name", default="brats_val")
    parser.add_argument("--model-id", choices=["S", "B", "M", "L"])
    parser.add_argument("--kernel-size", type=int, choices=[3, 5])
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--max-context-volumes", type=int)
    parser.add_argument("--max-volumes", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--slice-col", default="slice")
    parser.add_argument("--h5-image-key", default="image")
    parser.add_argument("--h5-mask-key", default="mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path(args.checkpoint).with_name(f"{args.split_name}_causal_metrics.json")
    )
    _save_json(metrics, output_json)
    print(metrics)
    print({"metrics_json": str(output_json)})


if __name__ == "__main__":
    main()
