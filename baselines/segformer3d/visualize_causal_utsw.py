from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.evaluate_causal_utsw import (
    _build_model,
    _config_value,
    _load_json,
    _make_dataset,
    _make_loader,
    _save_json,
)
from baselines.segformer3d.train_causal_utsw import build_context_bank
from baselines.segformer3d.train_utsw import _resolve_device


REGION_CHANNELS = {
    "NCR": (0,),
    "NCR_NET": (0,),
    "ED": (1,),
    "EDEMA": (1,),
    "ET": (2,),
    "WT": (0, 1, 2),
    "TC": (0, 2),
}

MODALITY_INDEX = {
    "flair": 0,
    "t1": 1,
    "t1ce": 2,
    "t2": 3,
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _region_channels(region: str) -> tuple[int, ...]:
    key = region.upper()
    if key not in REGION_CHANNELS:
        valid = ", ".join(sorted(REGION_CHANNELS))
        raise ValueError(f"Unknown region {region!r}. Choose one of: {valid}")
    return REGION_CHANNELS[key]


def region_probability_map(subregion_probs: Tensor, region: str) -> Tensor:
    """Convert BraTS subregion probabilities to a clinically named region map.

    Multi-channel regions use a noisy-or union. This is smoother than max for
    probability deltas while still respecting that WT and TC are unions of
    subregions.
    """

    channels = _region_channels(region)
    probs = subregion_probs.float().clamp(0.0, 1.0)
    selected = probs[list(channels)]
    if selected.shape[0] == 1:
        return selected[0]
    return 1.0 - torch.prod(1.0 - selected, dim=0)


def region_binary_mask(subregion_mask: Tensor, region: str) -> Tensor:
    channels = _region_channels(region)
    mask = subregion_mask.float().clamp(0.0, 1.0)
    return mask[list(channels)].amax(dim=0)


def _dice_score(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> float:
    prediction = prediction.float()
    target = target.float()
    intersection = torch.sum(prediction * target)
    denominator = torch.sum(prediction) + torch.sum(target)
    return float(((2.0 * intersection + eps) / (denominator + eps)).detach().cpu())


def select_slice_index(
    target_region: Tensor,
    factual_region: Tensor,
    adjusted_region: Tensor,
    effect_region: Tensor,
    threshold: float,
) -> int:
    """Pick the axial slice with the most interpretable causal content."""

    if target_region.ndim != 3:
        raise ValueError(f"Expected a 3D target region, got shape {tuple(target_region.shape)}")
    scores = target_region.sum(dim=(1, 2))
    if float(scores.max()) <= 0.0:
        factual_mask = factual_region >= float(threshold)
        adjusted_mask = adjusted_region >= float(threshold)
        scores = factual_mask.float().sum(dim=(1, 2)) + adjusted_mask.float().sum(dim=(1, 2))
    if float(scores.max()) <= 0.0:
        scores = effect_region.sum(dim=(1, 2))
    if float(scores.max()) <= 0.0:
        return int(target_region.shape[0] // 2)
    return int(torch.argmax(scores).item())


def _normalize_image_slice(image_slice: Tensor) -> np.ndarray:
    array = image_slice.detach().cpu().numpy().astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def _imshow_base(axis: plt.Axes, base: np.ndarray, title: str) -> None:
    axis.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def _overlay_binary(axis: plt.Axes, base: np.ndarray, mask: np.ndarray, title: str, color: str) -> None:
    _imshow_base(axis, base, title)
    masked = np.ma.masked_where(mask <= 0.5, mask)
    axis.imshow(masked, cmap=matplotlib.colors.ListedColormap([color]), alpha=0.42, vmin=0.0, vmax=1.0)
    if np.any(mask > 0.5):
        axis.contour(mask, levels=[0.5], colors="white", linewidths=0.8)


def _overlay_probability(
    figure: plt.Figure,
    axis: plt.Axes,
    base: np.ndarray,
    probability: np.ndarray,
    title: str,
    threshold: float,
    cmap: str,
) -> None:
    _imshow_base(axis, base, title)
    handle = axis.imshow(probability, cmap=cmap, alpha=0.62, vmin=0.0, vmax=1.0)
    if np.any(probability >= threshold):
        axis.contour(probability, levels=[threshold], colors="white", linewidths=0.8)
    figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.02)


def _overlay_effect(
    figure: plt.Figure,
    axis: plt.Axes,
    base: np.ndarray,
    effect: np.ndarray,
    title: str,
) -> None:
    _imshow_base(axis, base, title)
    vmax = float(np.max(effect))
    handle = axis.imshow(effect, cmap="magma", alpha=0.72, vmin=0.0, vmax=max(vmax, 1e-6))
    figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.02)


def _overlay_delta(
    figure: plt.Figure,
    axis: plt.Axes,
    base: np.ndarray,
    delta: np.ndarray,
    title: str,
) -> None:
    _imshow_base(axis, base, title)
    vmax = max(float(np.max(np.abs(delta))), 1e-6)
    handle = axis.imshow(delta, cmap="coolwarm", alpha=0.72, vmin=-vmax, vmax=vmax)
    figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.02)


def _safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def _case_id_from_args(splits: dict[str, list[str]], args: argparse.Namespace) -> str:
    split_case_ids = list(splits[args.split])
    if args.case_id:
        if args.case_id not in split_case_ids and not args.allow_case_outside_split:
            raise ValueError(
                f"Case {args.case_id!r} is not in the {args.split!r} split. "
                "Pass --allow-case-outside-split if you intentionally want that."
            )
        return str(args.case_id)
    if not 0 <= int(args.case_index) < len(split_case_ids):
        raise IndexError(f"--case-index must be in [0, {len(split_case_ids) - 1}], got {args.case_index}")
    return str(split_case_ids[int(args.case_index)])


@torch.no_grad()
def visualize(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    splits = checkpoint.get("splits")
    if splits is None:
        splits = _load_json(Path(args.checkpoint).with_name("splits.json"))

    case_id = _case_id_from_args(splits, args)
    data_root = Path(args.data_root or config.get("data_root", "data/brats/PKG - UTSW-Glioma/UTSW-Glioma"))
    eval_dataset = _make_dataset(data_root, [case_id], args, config)
    bank_dataset = _make_dataset(data_root, splits[args.context_split], args, config)
    if eval_dataset.metadata_encoder is None and not args.allow_missing_metadata:
        raise FileNotFoundError(
            "Causal visualization expects UTSW metadata proxies for the saved model layout. "
            "Pass --metadata-path or --allow-missing-metadata for a representation-only figure."
        )

    bank_batch_size = int(_config_value(args, config, "batch_size", 1))
    bank_loader = _make_loader(bank_dataset, batch_size=bank_batch_size, num_workers=args.num_workers)
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
    if context_bank is None or context_bank.numel() == 0:
        raise RuntimeError("Could not build a non-empty context bank for causal visualization.")
    bank_device = context_bank.to(device)

    sample = eval_dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
    logits = outputs["logits"]
    adjusted_logits = outputs.get("adjusted_logits")
    if not isinstance(logits, Tensor) or not isinstance(adjusted_logits, Tensor):
        raise TypeError("Causal visualization requires tensor 'logits' and 'adjusted_logits' outputs.")

    factual_probs = torch.sigmoid(logits[0].detach().cpu())
    adjusted_probs = torch.sigmoid(adjusted_logits[0].detach().cpu())
    target = sample["mask"].detach().cpu()

    target_region = region_binary_mask(target, args.region)
    factual_region = region_probability_map(factual_probs, args.region)
    adjusted_region = region_probability_map(adjusted_probs, args.region)
    effect_region = (factual_region - adjusted_region).abs()
    delta_region = adjusted_region - factual_region

    if args.slice_index is None:
        slice_index = select_slice_index(
            target_region=target_region,
            factual_region=factual_region,
            adjusted_region=adjusted_region,
            effect_region=effect_region,
            threshold=args.threshold,
        )
    else:
        slice_index = int(args.slice_index)
        depth = int(target_region.shape[0])
        if not 0 <= slice_index < depth:
            raise IndexError(f"--slice-index must be in [0, {depth - 1}], got {slice_index}")

    modality_index = MODALITY_INDEX[args.modality.lower()]
    base_slice = _normalize_image_slice(sample["image"][modality_index, slice_index])
    target_slice = target_region[slice_index].numpy()
    factual_slice = factual_region[slice_index].numpy()
    adjusted_slice = adjusted_region[slice_index].numpy()
    effect_slice = effect_region[slice_index].numpy()
    delta_slice = delta_region[slice_index].numpy()

    factual_mask = factual_region >= float(args.threshold)
    adjusted_mask = adjusted_region >= float(args.threshold)
    factual_dice = _dice_score(factual_mask, target_region)
    adjusted_dice = _dice_score(adjusted_mask, target_region)

    z_d = outputs.get("z_d")
    z_c = outputs.get("z_c")
    nearest_context_l2 = None
    if isinstance(z_c, Tensor):
        nearest_context_l2 = float(torch.cdist(z_c.detach(), bank_device).min().detach().cpu())

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    axes_flat = list(axes.ravel())
    _imshow_base(axes_flat[0], base_slice, f"{args.modality.upper()} MRI")
    _overlay_binary(axes_flat[1], base_slice, target_slice, f"Ground Truth {args.region.upper()}", "#2ca02c")
    _overlay_probability(fig, axes_flat[2], base_slice, factual_slice, "Factual Prediction", args.threshold, "viridis")
    _overlay_probability(fig, axes_flat[3], base_slice, adjusted_slice, "Context-Adjusted Prediction", args.threshold, "viridis")
    _overlay_effect(fig, axes_flat[4], base_slice, effect_slice, "|P factual - P adjusted|")
    _overlay_delta(fig, axes_flat[5], base_slice, delta_slice, "P adjusted - P factual")

    fig.suptitle(
        (
            f"Causal SegFormer3D visual explanation | case={case_id} split={args.split} "
            f"region={args.region.upper()} slice={slice_index} | "
            f"Dice factual={factual_dice:.3f}, adjusted={adjusted_dice:.3f} | "
            f"mean abs effect={float(effect_region.mean()):.6f}"
        ),
        fontsize=11,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{_safe_name(case_id)}_{args.region.upper()}_slice_{slice_index}"
    figure_path = Path(args.output_png) if args.output_png else output_dir / f"{stem}_causal_visual.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    summary_path = Path(args.output_json) if args.output_json else figure_path.with_suffix(".json")

    scm = default_utsw_scm()
    summary = {
        "figure": str(figure_path),
        "summary_json": str(summary_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "case_id": case_id,
        "split": args.split,
        "context_split": args.context_split,
        "region": args.region.upper(),
        "modality": args.modality.lower(),
        "slice_index": int(slice_index),
        "threshold": float(args.threshold),
        "context_bank_size": int(context_bank.shape[0]),
        "adjustment_contexts": int(args.adjustment_contexts),
        "factual_region_dice": factual_dice,
        "adjusted_region_dice": adjusted_dice,
        "adjusted_minus_factual_region_dice": adjusted_dice - factual_dice,
        "mean_abs_context_adjustment_effect": float(effect_region.mean()),
        "max_abs_context_adjustment_effect": float(effect_region.max()),
        "slice_mean_abs_context_adjustment_effect": float(effect_region[slice_index].mean()),
        "slice_max_abs_context_adjustment_effect": float(effect_region[slice_index].max()),
        "signed_context_adjustment_mean": float(delta_region.mean()),
        "target_foreground_voxels": float(target_region.sum()),
        "factual_foreground_voxels": float(factual_mask.float().sum()),
        "adjusted_foreground_voxels": float(adjusted_mask.float().sum()),
        "z_d_norm": None if not isinstance(z_d, Tensor) else float(z_d.detach().norm(dim=1).mean().cpu()),
        "z_c_norm": None if not isinstance(z_c, Tensor) else float(z_c.detach().norm(dim=1).mean().cpu()),
        "nearest_context_l2": nearest_context_l2,
        "causal_question": scm.question.query,
        "causal_estimand": scm.question.estimand,
        "causal_warning": scm.question.warning,
        "metadata_raw": _to_jsonable(sample.get("metadata_raw", {})),
    }

    _save_json(_to_jsonable(summary), summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export intervention-based causal visualizations for UTSW Causal SegFormer3D. "
            "The key map is the probability shift between factual and context-adjusted predictions."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--context-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--case-id")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--allow-case-outside-split", action="store_true")
    parser.add_argument("--region", default="ET", choices=["WT", "TC", "ET", "NCR", "NCR_NET", "ED", "EDEMA"])
    parser.add_argument("--modality", default="flair", choices=sorted(MODALITY_INDEX))
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data-root")
    parser.add_argument("--metadata-path")
    parser.add_argument("--model-size", choices=["tiny", "base"])
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-margin", type=int)
    parser.add_argument("--prefer-manual-seg", action="store_true", default=None)
    parser.add_argument("--use-ants-modalities", action="store_true", default=None)
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--output-dir", default="figures/causal_interpretability")
    parser.add_argument("--output-png")
    parser.add_argument("--output-json")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    summary = visualize(parse_args())
    print(json.dumps(_to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
