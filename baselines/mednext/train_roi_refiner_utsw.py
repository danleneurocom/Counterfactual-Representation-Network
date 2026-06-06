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
from baselines.mednext.evaluate_causal_utsw import (
    _build_model as _build_causal_model,
    _make_dataset as _make_causal_dataset,
)
from baselines.mednext.roi_refiner import (
    BBox3D,
    CausalRoiRefiner,
    bbox_from_mask,
    bbox_from_probabilities,
    crop_resize_3d,
    paste_resized_3d,
    scale_bbox,
    subregion_to_region_prob,
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


CORE_CHANNELS = (0, 2)


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
    coarse_logits: Tensor,
    global_bbox: BBox3D,
    roi_size: int,
) -> tuple[Tensor, Tensor, Tensor, BBox3D]:
    cropped_shape = tuple(int(value) for value in case["cropped_image"].shape[-3:])
    high_bbox = scale_bbox(global_bbox, source_shape=coarse_logits.shape[-3:], target_shape=cropped_shape)
    roi_image = crop_resize_3d(case["cropped_image"], high_bbox, roi_size, mode="trilinear")
    roi_mask = crop_resize_3d(case["cropped_mask"], high_bbox, roi_size, mode="nearest").clamp(0.0, 1.0)
    roi_coarse_logits = crop_resize_3d(coarse_logits, global_bbox, roi_size, mode="trilinear")
    return roi_image, roi_coarse_logits, roi_mask, high_bbox


def _training_bbox_from_source(case: dict[str, Tensor], coarse_logits: Tensor, args: argparse.Namespace) -> BBox3D:
    if args.train_roi_source == "target":
        return bbox_from_mask(case["global_mask"], margin=args.roi_margin)
    return bbox_from_probabilities(torch.sigmoid(coarse_logits), threshold=args.threshold, margin=args.roi_margin)


def _core_policy(policy: str) -> bool:
    return str(policy).startswith("core-")


def _apply_refinement_policy(coarse_logits: Tensor, refined_logits: Tensor, args: argparse.Namespace) -> Tensor:
    if args.roi_refinement_policy == "no-shrink":
        return torch.maximum(refined_logits, coarse_logits)
    if args.roi_refinement_policy == "core-only":
        output = coarse_logits.clone()
        output[:, CORE_CHANNELS] = refined_logits[:, CORE_CHANNELS]
        return output
    if args.roi_refinement_policy == "core-no-shrink":
        output = coarse_logits.clone()
        output[:, CORE_CHANNELS] = torch.maximum(refined_logits[:, CORE_CHANNELS], coarse_logits[:, CORE_CHANNELS])
        return output
    return refined_logits


def _roi_training_loss(refined_logits: Tensor, target: Tensor, coarse_logits: Tensor, args: argparse.Namespace) -> Tensor:
    if _core_policy(args.roi_refinement_policy):
        refined_core = refined_logits[:, CORE_CHANNELS]
        target_core = target[:, CORE_CHANNELS]
        coarse_core = coarse_logits[:, CORE_CHANNELS]
        return segmentation_loss(refined_core, target_core) + 0.25 * F.mse_loss(
            torch.sigmoid(refined_core),
            torch.sigmoid(coarse_core).detach(),
        )
    return segmentation_loss(refined_logits, target) + 0.25 * F.mse_loss(
        torch.sigmoid(refined_logits),
        torch.sigmoid(coarse_logits).detach(),
    )


def _paste_refined_roi(base_logits: Tensor, refined_roi: Tensor, bbox: BBox3D, args: argparse.Namespace) -> Tensor:
    if not _core_policy(args.roi_refinement_policy):
        return paste_resized_3d(base_logits, refined_roi, bbox)
    if not _core_intervention_trigger(base_logits, args):
        return base_logits
    slices = tuple(slice(start, stop) for start, stop in bbox)
    target_size = tuple(stop - start for start, stop in bbox)
    resized = F.interpolate(
        refined_roi.unsqueeze(0),
        size=target_size,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    output = base_logits.clone()
    output[(0, *slices)] = resized[(0, slice(None), slice(None), slice(None))]
    output[(2, *slices)] = resized[(2, slice(None), slice(None), slice(None))]
    return output


def _core_intervention_trigger(base_logits: Tensor, args: argparse.Namespace) -> bool:
    if args.roi_core_trigger == "all":
        return True
    region_prob = subregion_to_region_prob(torch.sigmoid(base_logits).unsqueeze(0)).squeeze(0)
    wt_voxels = int((region_prob[0] > float(args.threshold)).sum().item())
    tc_voxels = int((region_prob[1] > float(args.threshold)).sum().item())
    if wt_voxels < int(args.roi_core_trigger_min_wt_voxels):
        return False
    if args.roi_core_trigger == "missing-tc":
        return tc_voxels == 0
    if args.roi_core_trigger == "tiny-tc":
        return tc_voxels <= int(args.roi_core_trigger_max_tc_voxels)
    raise ValueError(f"Unknown ROI core trigger: {args.roi_core_trigger}")


def _target_tiny_core_case(target: Tensor, args: argparse.Namespace) -> bool:
    region = subregion_to_region_prob(target.unsqueeze(0).float()).squeeze(0)
    wt_voxels = int((region[0] > 0.5).sum().item())
    tc_voxels = int((region[1] > 0.5).sum().item())
    return (
        wt_voxels >= int(args.roi_core_trigger_min_wt_voxels)
        and tc_voxels <= int(args.roi_core_trigger_max_tc_voxels)
    )


def _keep_training_case(case: dict[str, Tensor], coarse_logits: Tensor | None, args: argparse.Namespace) -> bool:
    if args.train_roi_filter == "all":
        return True
    if args.train_roi_filter == "tiny-target-core":
        return _target_tiny_core_case(case["global_mask"], args)
    if args.train_roi_filter == "coarse-core-trigger":
        if coarse_logits is None:
            raise ValueError("coarse-core-trigger requires coarse logits.")
        return _core_intervention_trigger(coarse_logits, args)
    raise ValueError(f"Unknown train ROI filter: {args.train_roi_filter}")


def _load_coarse_model(
    checkpoint: dict[str, Any],
    root: Path,
    splits: dict[str, list[str]],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    if args.coarse_checkpoint_type == "baseline":
        eval_args = argparse.Namespace(model_id=None, kernel_size=None, base_channels=None)
        model = load_model_for_eval(checkpoint, eval_args)
    else:
        config = dict(checkpoint.get("config", {}))
        dataset_args = argparse.Namespace(
            volume_size=args.volume_size,
            crop_margin=args.crop_margin,
            prefer_manual_seg=args.prefer_manual_seg,
            use_ants_modalities=args.use_ants_modalities,
            metadata_path=args.metadata_path,
        )
        reference_ids = splits.get("train") or splits.get("val") or splits.get("test") or []
        if not reference_ids:
            raise ValueError("Cannot build causal coarse model without at least one split case id.")
        dataset = _make_causal_dataset(root, reference_ids[:1], dataset_args, config)
        model = _build_causal_model(checkpoint, dataset, dataset_args, config)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def _coarse_logits(model: torch.nn.Module, image: Tensor, device: torch.device, args: argparse.Namespace) -> Tensor:
    output = model(image.unsqueeze(0).to(device))
    if args.coarse_checkpoint_type == "causal":
        if not isinstance(output, dict) or not isinstance(output.get("logits"), Tensor):
            raise TypeError("Causal coarse checkpoint must return a dict with tensor key 'logits'.")
        return output["logits"].squeeze(0).detach()
    return main_logits(output).squeeze(0).detach()


def _train_one_epoch(
    refiner: CausalRoiRefiner,
    coarse_model: torch.nn.Module,
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
    trained_steps = 0
    for index, case_id in enumerate(tqdm(case_ids, desc="roi-refiner-train", leave=False), start=1):
        case = _load_case(root, case_id, args)
        if args.train_roi_filter == "tiny-target-core" and not _keep_training_case(case, None, args):
            continue
        coarse_logits = _coarse_logits(coarse_model, case["global_image"], device, args).cpu()
        if args.train_roi_filter == "coarse-core-trigger" and not _keep_training_case(case, coarse_logits, args):
            continue
        global_bbox = _training_bbox_from_source(case, coarse_logits, args)
        roi_image, roi_coarse_logits, roi_mask, _ = _roi_from_bbox(case, coarse_logits, global_bbox, args.roi_size)

        optimizer.zero_grad(set_to_none=True)
        raw_refined_roi = refiner(
            roi_image.unsqueeze(0).to(device),
            roi_coarse_logits.unsqueeze(0).to(device),
        )
        roi_coarse_logits_device = roi_coarse_logits.unsqueeze(0).to(device)
        refined_roi = _apply_refinement_policy(roi_coarse_logits_device, raw_refined_roi, args)
        target = roi_mask.unsqueeze(0).to(device)
        loss = _roi_training_loss(refined_roi, target, roi_coarse_logits_device, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        metric_items.append(brats_region_metrics(refined_roi.detach().cpu(), roi_mask.unsqueeze(0), threshold=args.threshold))
        trained_steps += 1
        if args.max_train_batches is not None and trained_steps >= args.max_train_batches:
            break
    metrics = _average_metric_dicts(metric_items)
    metrics["loss"] = _mean(losses)
    metrics["train/roi_updates"] = float(trained_steps)
    return metrics


@torch.no_grad()
def _eval_epoch(
    refiner: CausalRoiRefiner,
    coarse_model: torch.nn.Module,
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
    core_trigger_count = 0
    for index, case_id in enumerate(tqdm(case_ids, desc="roi-refiner-val", leave=False), start=1):
        case = _load_case(root, case_id, args)
        global_image = case["global_image"]
        global_mask = case["global_mask"]
        coarse_logits = _coarse_logits(coarse_model, global_image, device, args).cpu()
        baseline_items.append(brats_region_metrics(coarse_logits.unsqueeze(0), global_mask.unsqueeze(0), threshold=args.threshold))

        pred_bbox = bbox_from_probabilities(torch.sigmoid(coarse_logits), threshold=args.threshold, margin=args.roi_margin)
        roi_image, roi_coarse_logits, roi_mask, _ = _roi_from_bbox(case, coarse_logits, pred_bbox, args.roi_size)
        roi_coarse_logits_device = roi_coarse_logits.unsqueeze(0).to(device)
        raw_refined_roi = refiner(roi_image.unsqueeze(0).to(device), roi_coarse_logits_device)
        refined_roi = _apply_refinement_policy(roi_coarse_logits_device, raw_refined_roi, args).detach().cpu().squeeze(0)
        if _core_policy(args.roi_refinement_policy) and _core_intervention_trigger(coarse_logits, args):
            core_trigger_count += 1
        full_refined = _paste_refined_roi(coarse_logits, refined_roi, pred_bbox, args)

        losses.append(float(segmentation_loss(full_refined.unsqueeze(0), global_mask.unsqueeze(0)).detach().cpu()))
        refined_items.append(brats_region_metrics(full_refined.unsqueeze(0), global_mask.unsqueeze(0), threshold=args.threshold))
        roi_items.append(brats_region_metrics(refined_roi.unsqueeze(0), roi_mask.unsqueeze(0), threshold=args.threshold))
        if args.max_val_batches is not None and index >= args.max_val_batches:
            break
    metrics: dict[str, float] = {"loss": _mean(losses)}
    metrics.update({f"baseline/{key}": value for key, value in _average_metric_dicts(baseline_items).items()})
    metrics.update({f"refined/{key}": value for key, value in _average_metric_dicts(refined_items).items()})
    metrics.update({f"roi/{key}": value for key, value in _average_metric_dicts(roi_items).items()})
    if _core_policy(args.roi_refinement_policy):
        metrics["roi/core_trigger_count"] = float(core_trigger_count)
    if "baseline/brats/mean_dice" in metrics and "refined/brats/mean_dice" in metrics:
        metrics["refined_minus_baseline/brats/mean_dice"] = metrics["refined/brats/mean_dice"] - metrics["baseline/brats/mean_dice"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a high-resolution causal ROI refiner on UTSW.")
    parser.add_argument("--baseline-checkpoint", default="runs/mednext_utsw_s_k3/best.pt")
    parser.add_argument("--coarse-checkpoint-type", choices=("baseline", "causal"), default="baseline")
    parser.add_argument("--data-root", default="data/brats/PKG - UTSW-Glioma/UTSW-Glioma")
    parser.add_argument("--metadata-path")
    parser.add_argument("--splits-json")
    parser.add_argument("--output-dir", default="runs/mednext_utsw_roi_refiner")
    parser.add_argument("--volume-size", type=int, default=64)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--roi-margin", type=int, default=6)
    parser.add_argument("--train-roi-source", choices=("prediction", "target"), default="prediction")
    parser.add_argument(
        "--train-roi-filter",
        choices=("all", "tiny-target-core", "coarse-core-trigger"),
        default="all",
    )
    parser.add_argument(
        "--roi-refinement-policy",
        choices=("signed", "no-shrink", "core-only", "core-no-shrink"),
        default="signed",
    )
    parser.add_argument("--roi-core-trigger", choices=("all", "missing-tc", "tiny-tc"), default="all")
    parser.add_argument("--roi-core-trigger-max-tc-voxels", type=int, default=32)
    parser.add_argument("--roi-core-trigger-min-wt-voxels", type=int, default=64)
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
    coarse_model = _load_coarse_model(checkpoint, root, splits, args, device)

    refiner = CausalRoiRefiner().to(device)
    optimizer = AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_ids = list(splits["train"])
        val_ids = list(splits["val"])
        train_metrics = _train_one_epoch(refiner, coarse_model, root, train_ids, optimizer, device, args)
        val_metrics = _eval_epoch(refiner, coarse_model, root, val_ids, device, args)
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
    final_metrics = _eval_epoch(refiner, coarse_model, root, list(splits["val"]), device, args)
    save_json(final_metrics, output_dir / "val_roi_refiner_metrics.json")
    print({"best_val_refined_brats_mean_dice": best_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
