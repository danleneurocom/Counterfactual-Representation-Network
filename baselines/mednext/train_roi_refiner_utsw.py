from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm

from baselines.mednext.common import load_model_for_eval, main_logits, save_json, segmentation_loss, _resolve_device
from baselines.mednext.roi_refiner import (
    BBox3D,
    CausalRoiRefiner,
    bbox_from_mask,
    bbox_from_probabilities,
    crop_resize_3d,
    paste_resized_3d,
    scale_bbox,
)
from baselines.segformer3d.data.utsw import (
    UTSW_MODALITIES,
    _crop_to_foreground,
    _find_modality,
    _find_segmentation,
    _load_nifti,
    _normalize_mri,
    _resize_volume,
    _subregion_mask,
    _to_depth_first,
)
from baselines.segformer3d.train_utsw import _average_metric_dicts, _case_ids, _make_splits, _mean
from crn.metrics import brats_region_metrics


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_case(root: Path, case_id: str, args: argparse.Namespace) -> dict[str, Tensor]:
    case_dir = root / case_id
    modality_paths = [_find_modality(case_dir, modality, args.use_ants_modalities) for modality in UTSW_MODALITIES]
    seg_path = _find_segmentation(case_dir, args.prefer_manual_seg)
    volumes = [_normalize_mri(_to_depth_first(_load_nifti(path))) for path in modality_paths]
    image = np.stack(volumes, axis=0)
    segmentation = _to_depth_first(_load_nifti(seg_path))
    if segmentation.shape != image.shape[1:]:
        fallback = case_dir / "tumorseg_FeTS.nii.gz"
        if fallback.exists() and fallback != seg_path:
            segmentation = _to_depth_first(_load_nifti(fallback))
    if segmentation.shape != image.shape[1:]:
        raise ValueError(f"Segmentation shape {segmentation.shape} does not match image shape {image.shape[1:]} for {case_id}")
    mask = _subregion_mask(segmentation)
    cropped_image, cropped_mask = _crop_to_foreground(image, mask, args.crop_margin)
    global_image, global_mask = _resize_volume(cropped_image, cropped_mask, args.volume_size)
    return {
        "global_image": global_image.clamp(-6.0, 6.0),
        "global_mask": global_mask.clamp(0.0, 1.0),
        "cropped_image": torch.from_numpy(np.ascontiguousarray(cropped_image)).float(),
        "cropped_mask": torch.from_numpy(np.ascontiguousarray(cropped_mask)).float().clamp(0.0, 1.0),
    }


def _roi_from_bbox(
    case: dict[str, Tensor],
    baseline_logits: Tensor,
    global_bbox: BBox3D,
    roi_size: int,
) -> tuple[Tensor, Tensor, Tensor, BBox3D]:
    cropped_shape = tuple(int(value) for value in case["cropped_image"].shape[-3:])
    high_bbox = scale_bbox(global_bbox, source_shape=baseline_logits.shape[-3:], target_shape=cropped_shape)
    roi_image = crop_resize_3d(case["cropped_image"], high_bbox, roi_size, mode="trilinear")
    roi_mask = crop_resize_3d(case["cropped_mask"], high_bbox, roi_size, mode="nearest").clamp(0.0, 1.0)
    coarse_logits = crop_resize_3d(baseline_logits, global_bbox, roi_size, mode="trilinear")
    return roi_image, coarse_logits, roi_mask, high_bbox


def _training_bbox_from_source(case: dict[str, Tensor], baseline_logits: Tensor, args: argparse.Namespace) -> BBox3D:
    if args.train_roi_source == "target":
        return bbox_from_mask(case["global_mask"], margin=args.roi_margin)
    return bbox_from_probabilities(torch.sigmoid(baseline_logits), threshold=args.threshold, margin=args.roi_margin)


def _apply_refinement_policy(coarse_logits: Tensor, refined_logits: Tensor, args: argparse.Namespace) -> Tensor:
    if args.roi_refinement_policy == "no-shrink":
        return torch.maximum(refined_logits, coarse_logits)
    return refined_logits


@torch.no_grad()
def _baseline_logits(model: torch.nn.Module, image: Tensor, device: torch.device) -> Tensor:
    output = model(image.unsqueeze(0).to(device))
    return main_logits(output).squeeze(0).detach()


def _train_one_epoch(
    refiner: CausalRoiRefiner,
    baseline: torch.nn.Module,
    root: Path,
    case_ids: list[str],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    refiner.train()
    random.shuffle(case_ids)
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    for index, case_id in enumerate(tqdm(case_ids, desc="roi-refiner-train", leave=False), start=1):
        case = _load_case(root, case_id, args)
        baseline_logits = _baseline_logits(baseline, case["global_image"], device).cpu()
        global_bbox = _training_bbox_from_source(case, baseline_logits, args)
        roi_image, coarse_logits, roi_mask, _ = _roi_from_bbox(case, baseline_logits, global_bbox, args.roi_size)

        optimizer.zero_grad(set_to_none=True)
        raw_refined_roi = refiner(
            roi_image.unsqueeze(0).to(device),
            coarse_logits.unsqueeze(0).to(device),
        )
        coarse_logits_device = coarse_logits.unsqueeze(0).to(device)
        refined_roi = _apply_refinement_policy(coarse_logits_device, raw_refined_roi, args)
        target = roi_mask.unsqueeze(0).to(device)
        loss = segmentation_loss(refined_roi, target) + 0.25 * F.mse_loss(torch.sigmoid(refined_roi), torch.sigmoid(coarse_logits_device).detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(refined_roi.detach().cpu(), roi_mask.unsqueeze(0), threshold=args.threshold))
        if args.max_train_batches is not None and index >= args.max_train_batches:
            break
    metrics = _average_metric_dicts(metric_items)
    metrics["loss"] = _mean(losses)
    return metrics


@torch.no_grad()
def _eval_epoch(
    refiner: CausalRoiRefiner,
    baseline: torch.nn.Module,
    root: Path,
    case_ids: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    refiner.eval()
    baseline_items: list[dict[str, float]] = []
    refined_items: list[dict[str, float]] = []
    roi_items: list[dict[str, float]] = []
    losses: list[float] = []
    for index, case_id in enumerate(tqdm(case_ids, desc="roi-refiner-val", leave=False), start=1):
        case = _load_case(root, case_id, args)
        global_image = case["global_image"]
        global_mask = case["global_mask"]
        baseline_logits = _baseline_logits(baseline, global_image, device).cpu()
        baseline_items.append(brats_region_metrics(baseline_logits.unsqueeze(0), global_mask.unsqueeze(0), threshold=args.threshold))

        pred_bbox = bbox_from_probabilities(torch.sigmoid(baseline_logits), threshold=args.threshold, margin=args.roi_margin)
        roi_image, coarse_logits, roi_mask, _ = _roi_from_bbox(case, baseline_logits, pred_bbox, args.roi_size)
        coarse_logits_device = coarse_logits.unsqueeze(0).to(device)
        raw_refined_roi = refiner(roi_image.unsqueeze(0).to(device), coarse_logits_device)
        refined_roi = _apply_refinement_policy(coarse_logits_device, raw_refined_roi, args).detach().cpu().squeeze(0)
        full_refined = paste_resized_3d(baseline_logits, refined_roi, pred_bbox)

        losses.append(float(segmentation_loss(full_refined.unsqueeze(0), global_mask.unsqueeze(0)).detach().cpu()))
        refined_items.append(brats_region_metrics(full_refined.unsqueeze(0), global_mask.unsqueeze(0), threshold=args.threshold))
        roi_items.append(brats_region_metrics(refined_roi.unsqueeze(0), roi_mask.unsqueeze(0), threshold=args.threshold))
        if args.max_val_batches is not None and index >= args.max_val_batches:
            break
    metrics: dict[str, float] = {"loss": _mean(losses)}
    metrics.update({f"baseline/{key}": value for key, value in _average_metric_dicts(baseline_items).items()})
    metrics.update({f"refined/{key}": value for key, value in _average_metric_dicts(refined_items).items()})
    metrics.update({f"roi/{key}": value for key, value in _average_metric_dicts(roi_items).items()})
    if "baseline/brats/mean_dice" in metrics and "refined/brats/mean_dice" in metrics:
        metrics["refined_minus_baseline/brats/mean_dice"] = metrics["refined/brats/mean_dice"] - metrics["baseline/brats/mean_dice"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a high-resolution causal ROI refiner on UTSW.")
    parser.add_argument("--baseline-checkpoint", default="runs/mednext_utsw_s_k3/best.pt")
    parser.add_argument("--data-root", default="data/brats/PKG - UTSW-Glioma/UTSW-Glioma")
    parser.add_argument("--splits-json")
    parser.add_argument("--output-dir", default="runs/mednext_utsw_roi_refiner")
    parser.add_argument("--volume-size", type=int, default=64)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--roi-margin", type=int, default=6)
    parser.add_argument("--train-roi-source", choices=("prediction", "target"), default="prediction")
    parser.add_argument("--roi-refinement-policy", choices=("signed", "no-shrink"), default="signed")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--prefer-manual-seg", action="store_true")
    parser.add_argument("--use-ants-modalities", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")
    if args.splits_json:
        splits = _load_json(Path(args.splits_json))
    else:
        splits = checkpoint.get("splits") or _make_splits(_case_ids(root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)
    save_json(splits, output_dir / "splits.json")
    save_json(vars(args), output_dir / "config.json")

    device = _resolve_device(args.device)
    eval_args = argparse.Namespace(model_id=None, kernel_size=None, base_channels=None)
    baseline = load_model_for_eval(checkpoint, eval_args).to(device)
    baseline.eval()
    for parameter in baseline.parameters():
        parameter.requires_grad = False

    refiner = CausalRoiRefiner().to(device)
    optimizer = AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_ids = list(splits["train"])
        val_ids = list(splits["val"])
        train_metrics = _train_one_epoch(refiner, baseline, root, train_ids, optimizer, device, args)
        val_metrics = _eval_epoch(refiner, baseline, root, val_ids, device, args)
        save_json({"epoch": epoch, "train": train_metrics, "val": val_metrics}, output_dir / f"epoch_{epoch:03d}.json")
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss"),
                "val_baseline_dice": val_metrics.get("baseline/brats/mean_dice"),
                "val_refined_dice": val_metrics.get("refined/brats/mean_dice"),
                "delta": val_metrics.get("refined_minus_baseline/brats/mean_dice"),
            }
        )
        checkpoint_data: dict[str, Any] = {
            "epoch": epoch,
            "model": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "splits": splits,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint_data, output_dir / "last.pt")
        monitor = float(val_metrics.get("refined/brats/mean_dice", float("-inf")))
        if monitor > best_dice:
            best_dice = monitor
            torch.save(checkpoint_data, output_dir / "best.pt")
    final_metrics = _eval_epoch(refiner, baseline, root, list(splits["val"]), device, args)
    save_json(final_metrics, output_dir / "val_roi_refiner_metrics.json")
    print({"best_val_refined_brats_mean_dice": best_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
