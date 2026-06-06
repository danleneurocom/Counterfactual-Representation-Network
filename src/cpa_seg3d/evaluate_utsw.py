from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_causal_utsw import build_context_bank, _metadata_dims, _prefix_metrics, _typed_proxy_loss
from baselines.segformer3d.train_utsw import _average_metric_dicts, _mean, _resolve_device
from cpa_seg3d import boundary_targets_from_subregions, build_cpa_seg3d, region_targets_from_subregions
from crn.metrics import brats_region_metrics, brats_volume_metrics_from_probs, multilabel_segmentation_metrics


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _config_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    return config.get(name, default) if value is None else value


def _make_dataset(root: Path, case_ids: list[str], args: argparse.Namespace, config: dict[str, Any]) -> UTSWGliomaDataset:
    return UTSWGliomaDataset(
        root=root,
        volume_size=int(_config_value(args, config, "volume_size", 128)),
        case_ids=case_ids,
        crop_margin=int(_config_value(args, config, "crop_margin", 8)),
        prefer_manual_seg=bool(_config_value(args, config, "prefer_manual_seg", False)),
        use_ants_modalities=bool(_config_value(args, config, "use_ants_modalities", False)),
        metadata_path=_config_value(args, config, "metadata_path"),
        include_metadata=True,
    )


def _make_loader(dataset: UTSWGliomaDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _proxy_dims(checkpoint: dict[str, Any], dataset: UTSWGliomaDataset) -> dict[str, int]:
    fallback = _metadata_dims(dataset)
    saved = checkpoint.get("proxy_dims") or {}
    return {
        "context_proxy_dim": int(saved.get("context_proxy_dim", fallback["context_proxy_dim"])),
        "disease_proxy_dim": int(saved.get("disease_proxy_dim", fallback["disease_proxy_dim"])),
        "annotation_proxy_dim": int(saved.get("annotation_proxy_dim", fallback["annotation_proxy_dim"])),
    }


def _decoder_variant(checkpoint: dict[str, Any], args: argparse.Namespace, config: dict[str, Any]) -> str:
    requested = getattr(args, "decoder_variant", None)
    if requested is not None:
        return str(requested)
    configured = config.get("decoder_variant")
    if configured is not None:
        return str(configured)
    state_dict = checkpoint.get("model", {})
    if any(str(key).startswith("decoder.full_refine.") for key in state_dict):
        return "unet"
    return "lite"


def _build_model(checkpoint: dict[str, Any], dataset: UTSWGliomaDataset, args: argparse.Namespace, config: dict[str, Any]):
    model = build_cpa_seg3d(
        model_size=str(_config_value(args, config, "model_size", "base")),
        latent_dim=int(_config_value(args, config, "latent_dim", 128)),
        num_classes=3,
        decoder_variant=_decoder_variant(checkpoint, args, config),
        **_proxy_dims(checkpoint, dataset),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def _dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _bce_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + _dice_loss(logits, target)


def _volume_metrics(logits: Tensor, target: Tensor, threshold: float) -> list[dict[str, float]]:
    probs_cpu = torch.sigmoid(logits.detach().cpu())
    target_cpu = target.detach().cpu()
    metrics: list[dict[str, float]] = []
    for row in range(probs_cpu.shape[0]):
        probs_volume = probs_cpu[row].permute(1, 0, 2, 3).contiguous()
        target_volume = target_cpu[row].permute(1, 0, 2, 3).contiguous()
        metrics.append(
            brats_volume_metrics_from_probs(
                probs_volume,
                target_volume,
                threshold=threshold,
                channel_names=["ncr_net", "edema", "enhancing_tumor"],
                region_channels={"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]},
            )
        )
    return metrics


def _proxy_layout(checkpoint: dict[str, Any], dataset: UTSWGliomaDataset) -> dict[str, list[dict[str, Any]]] | None:
    saved = checkpoint.get("proxy_layout")
    if saved:
        return saved
    if dataset.metadata_encoder is None:
        return None
    return dataset.metadata_encoder.proxy_layout()


def _proxy_losses(
    outputs: dict[str, Tensor | tuple[Tensor, ...] | list[Tensor]],
    batch: dict[str, Any],
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, float]:
    pairs = (
        ("context_proxy", "context_proxy_logits", "observed_context", None if proxy_layout is None else proxy_layout.get("context")),
        ("disease_proxy", "disease_proxy_logits", "observed_disease", None if proxy_layout is None else proxy_layout.get("disease")),
        ("annotation_proxy", "annotation_proxy_logits", "observed_annotation", None if proxy_layout is None else proxy_layout.get("annotation")),
    )
    losses: dict[str, float] = {}
    for name, output_key, target_key, layout in pairs:
        if output_key not in outputs or target_key not in batch:
            continue
        prediction = outputs[output_key]
        if not isinstance(prediction, Tensor):
            continue
        target = batch[target_key].to(device=prediction.device, dtype=prediction.dtype)
        typed = _typed_proxy_loss(prediction, target, layout)
        if typed is not None:
            losses[f"proxy/{name}_loss"] = float(typed.detach().cpu())
        losses[f"proxy/{name}_mse"] = float(F.mse_loss(prediction, target).detach().cpu())
    return losses


def _context_overlap(z_c: Tensor, context_bank: Tensor | None) -> dict[str, list[float]]:
    if context_bank is None or context_bank.numel() == 0:
        return {}
    bank = context_bank.to(device=z_c.device, dtype=z_c.dtype)
    distances = torch.cdist(z_c.detach(), bank)
    nearest = distances.min(dim=1).values.detach().cpu()
    return {"overlap/nearest_context_l2": [float(value) for value in nearest]}


def _region_head_metrics(logits: Tensor, target: Tensor, threshold: float) -> dict[str, float]:
    return multilabel_segmentation_metrics(
        logits.detach().cpu(),
        target.detach().cpu(),
        threshold=threshold,
        channel_names=["WT", "TC", "ET"],
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    splits = checkpoint.get("splits")
    if splits is None:
        splits = _load_json(Path(args.checkpoint).with_name("splits.json"))

    data_root = Path(args.data_root or config.get("data_root", "data/brats/PKG - UTSW-Glioma/UTSW-Glioma"))
    batch_size = int(_config_value(args, config, "batch_size", 1))
    eval_dataset = _make_dataset(data_root, splits[args.split], args, config)
    bank_dataset = _make_dataset(data_root, splits[args.context_split], args, config)
    if eval_dataset.metadata_encoder is None and not args.allow_missing_metadata:
        raise FileNotFoundError(
            "CPA-Seg3D evaluation needs metadata proxies for proxy and overlap diagnostics. "
            "Pass --metadata-path or --allow-missing-metadata for a representation-only ablation."
        )

    eval_loader = _make_loader(eval_dataset, batch_size=batch_size, num_workers=args.num_workers)
    bank_loader = _make_loader(bank_dataset, batch_size=batch_size, num_workers=args.num_workers)
    proxy_layout = _proxy_layout(checkpoint, eval_dataset)

    device = _resolve_device(args.device)
    model = _build_model(checkpoint, eval_dataset, args, config).to(device)
    model.eval()
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
    region_head_metric_items: list[dict[str, float]] = []
    factual_volume_metrics: list[dict[str, float]] = []
    adjusted_volume_metrics: list[dict[str, float]] = []
    proxy_metric_items: list[dict[str, float]] = []
    context_shifts: list[float] = []
    nearest_context_distances: list[float] = []

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"cpa-eval:{args.split}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        region_target = region_targets_from_subregions(target)
        boundary_target = boundary_targets_from_subregions(target)
        outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
        logits = outputs["logits"]
        region_logits = outputs["region_logits"]
        boundary_logits = outputs["boundary_logits"]
        if not isinstance(logits, Tensor) or not isinstance(region_logits, Tensor) or not isinstance(boundary_logits, Tensor):
            raise TypeError("CPA-Seg3D logits, region_logits, and boundary_logits must be tensors.")

        factual_loss = _bce_dice_loss(logits, target) + 0.3 * _bce_dice_loss(region_logits, region_target)
        factual_loss = factual_loss + 0.1 * _bce_dice_loss(boundary_logits, boundary_target)
        factual_losses.append(float(factual_loss.detach().cpu()))
        factual_batch_metrics.append(brats_region_metrics(logits.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
        region_head_metric_items.append(_region_head_metrics(region_logits, region_target, args.threshold))
        factual_volume_metrics.extend(_volume_metrics(logits, target, args.threshold))
        proxy_metric_items.append(_proxy_losses(outputs, batch, proxy_layout))

        z_c = outputs["z_c"]
        if isinstance(z_c, Tensor):
            nearest_context_distances.extend(_context_overlap(z_c, bank_device).get("overlap/nearest_context_l2", []))

        adjusted = outputs.get("adjusted_logits")
        if isinstance(adjusted, Tensor):
            adjusted_losses.append(float(_bce_dice_loss(adjusted, target).detach().cpu()))
            adjusted_batch_metrics.append(brats_region_metrics(adjusted.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
            adjusted_volume_metrics.extend(_volume_metrics(adjusted, target, args.threshold))
            context_shifts.append(float((torch.sigmoid(logits) - torch.sigmoid(adjusted)).abs().mean().detach().cpu()))

        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    scm = default_utsw_scm()
    metrics: dict[str, Any] = {
        "method": "CPA-Seg3D",
        "causal_question": scm.question.query,
        "causal_estimand": scm.question.estimand,
        "causal_warning": scm.question.warning,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "context_split": args.context_split,
        "threshold": float(args.threshold),
        "num_cases": float(len(factual_volume_metrics)),
        "context_bank_size": float(0 if context_bank is None else context_bank.shape[0]),
        "factual/loss": _mean(factual_losses),
    }
    metrics.update(_average_metric_dicts(factual_batch_metrics))
    metrics.update({f"volume/{key}": value for key, value in _average_metric_dicts(factual_volume_metrics).items()})
    metrics.update(_prefix_metrics(_average_metric_dicts(region_head_metric_items), "region_head"))
    metrics.update(_average_metric_dicts(proxy_metric_items))
    if adjusted_losses:
        metrics["adjusted/loss"] = _mean(adjusted_losses)
        metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_batch_metrics), "adjusted"))
        metrics.update({f"adjusted/volume/{key}": value for key, value in _average_metric_dicts(adjusted_volume_metrics).items()})
        metrics["intervention/context_adjustment_mean_abs_prob_shift"] = _mean(context_shifts)
        factual_dice = float(metrics.get("brats/mean_dice", float("nan")))
        adjusted_dice = float(metrics.get("adjusted/brats/mean_dice", float("nan")))
        metrics["intervention/adjusted_minus_factual_mean_dice"] = adjusted_dice - factual_dice
    if nearest_context_distances:
        distances = torch.tensor(nearest_context_distances, dtype=torch.float32)
        metrics["overlap/nearest_context_l2_mean"] = float(distances.mean())
        metrics["overlap/nearest_context_l2_max"] = float(distances.max())
        metrics["overlap/nearest_context_l2_p90"] = float(torch.quantile(distances, 0.9))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CPA-Seg3D with factual, adjusted, region-head, proxy, and overlap metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--context-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--data-root")
    parser.add_argument("--metadata-path")
    parser.add_argument("--output-json")
    parser.add_argument("--model-size", choices=["tiny", "base"])
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--decoder-variant", choices=["lite", "unet"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-margin", type=int)
    parser.add_argument("--prefer-manual-seg", action="store_true", default=None)
    parser.add_argument("--use-ants-modalities", action="store_true", default=None)
    parser.add_argument("--allow-missing-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    output_json = Path(args.output_json) if args.output_json else Path(args.checkpoint).with_name(f"{args.split}_cpa_metrics.json")
    _save_json(metrics, output_json)
    print(metrics)
    print({"metrics_json": str(output_json)})


if __name__ == "__main__":
    main()
