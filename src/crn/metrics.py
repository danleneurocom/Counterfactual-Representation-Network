from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, generate_binary_structure, label
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
import torch
from torch import Tensor


def _as_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _threshold_predictions(logits: Tensor, target: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = target.float()
    return pred, target


def _segmentation_stats(pred: Tensor, target: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dims = tuple(range(2, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)
    return dice, iou, precision, recall


def _volume_layout(tensor: Tensor) -> Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected a volume tensor with shape [depth, channels, height, width], got {tuple(tensor.shape)}")
    return tensor.permute(1, 0, 2, 3).unsqueeze(0)


def _default_hd95_empty_value(shape: tuple[int, ...], voxel_spacing: tuple[float, ...] | None) -> float:
    spacing = np.asarray(voxel_spacing if voxel_spacing is not None else (1.0,) * len(shape), dtype=np.float64)
    extents = np.maximum(np.asarray(shape, dtype=np.float64) - 1.0, 0.0) * spacing
    return float(np.linalg.norm(extents))


def hd95_from_masks(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    voxel_spacing: tuple[float, ...] | None = None,
) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)

    pred_has_foreground = bool(pred.any())
    target_has_foreground = bool(target.any())
    if not pred_has_foreground and not target_has_foreground:
        return 0.0
    if not pred_has_foreground or not target_has_foreground:
        return _default_hd95_empty_value(pred.shape, voxel_spacing)

    structure = generate_binary_structure(pred.ndim, 1)
    pred_surface = np.logical_xor(pred, binary_erosion(pred, structure=structure, border_value=0))
    target_surface = np.logical_xor(target, binary_erosion(target, structure=structure, border_value=0))
    if not pred_surface.any():
        pred_surface = pred
    if not target_surface.any():
        target_surface = target

    pred_to_target = distance_transform_edt(~target_surface, sampling=voxel_spacing)[pred_surface]
    target_to_pred = distance_transform_edt(~pred_surface, sampling=voxel_spacing)[target_surface]
    surface_distances = np.concatenate([pred_to_target, target_to_pred]).astype(np.float64, copy=False)
    return float(np.percentile(surface_distances, 95.0))


def postprocess_binary_volume(
    mask: np.ndarray,
    min_component_size: int = 0,
    fill_holes: bool = False,
    keep_largest: bool = False,
    connectivity: int = 1,
) -> np.ndarray:
    processed = np.asarray(mask, dtype=bool)
    if processed.ndim < 2:
        raise ValueError(f"Expected at least 2 spatial dimensions, got shape {processed.shape}")
    structure = generate_binary_structure(processed.ndim, connectivity)

    if fill_holes and processed.any():
        processed = binary_fill_holes(processed)

    labeled, num_components = label(processed, structure=structure)
    if num_components == 0:
        return processed

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0

    if keep_largest and component_sizes.max(initial=0) > 0:
        keep_labels = [int(component_sizes.argmax())]
        processed = np.isin(labeled, keep_labels)
    elif min_component_size > 0:
        keep_labels = np.flatnonzero(component_sizes >= int(min_component_size))
        processed = np.isin(labeled, keep_labels) if keep_labels.size else np.zeros_like(processed, dtype=bool)

    if fill_holes and processed.any():
        processed = binary_fill_holes(processed)
    return processed.astype(bool, copy=False)


def binary_volume_metrics_from_masks(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    voxel_spacing: tuple[float, ...] | None = None,
    eps: float = 1e-6,
) -> dict[str, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    intersection = float(np.logical_and(pred, target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = pred_sum + target_sum - intersection

    return {
        "dice": float((2.0 * intersection + eps) / (pred_sum + target_sum + eps)),
        "iou": float((intersection + eps) / (union + eps)),
        "precision": float((intersection + eps) / (pred_sum + eps)),
        "recall": float((intersection + eps) / (target_sum + eps)),
        "pred_foreground_ratio": float(pred.mean()),
        "target_foreground_ratio": float(target.mean()),
        "hd95": hd95_from_masks(pred, target, voxel_spacing),
    }


def brats_region_arrays_from_volume(
    volume: Tensor | np.ndarray,
    region_channels: dict[str, list[int]] | None = None,
) -> dict[str, np.ndarray]:
    array = volume.detach().cpu().numpy() if isinstance(volume, Tensor) else np.asarray(volume)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Expected [depth, channels, height, width], got {array.shape}")
    region_channels = region_channels or {"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]}
    return {
        region_name: array[:, channels].max(axis=1)
        for region_name, channels in region_channels.items()
    }


def binary_segmentation_metrics(logits: Tensor, target: Tensor, threshold: float = 0.5, eps: float = 1e-6) -> dict[str, float]:
    pred, target = _threshold_predictions(logits, target, threshold)
    dice, iou, precision, recall = _segmentation_stats(pred, target, eps)

    return {
        "seg/slice_dice_mean": float(dice.mean().item()),
        "seg/slice_iou_mean": float(iou.mean().item()),
        "seg/slice_precision_mean": float(precision.mean().item()),
        "seg/slice_recall_mean": float(recall.mean().item()),
        "seg/pred_foreground_ratio": float(pred.mean().item()),
        "seg/target_foreground_ratio": float(target.mean().item()),
    }


def multilabel_segmentation_metrics(
    logits: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    channel_names: list[str] | None = None,
    eps: float = 1e-6,
) -> dict[str, float]:
    pred, target = _threshold_predictions(logits, target, threshold)
    dice, iou, precision, recall = _segmentation_stats(pred, target, eps)
    metrics = {
        "seg/macro_dice_mean": float(dice.mean().item()),
        "seg/macro_iou_mean": float(iou.mean().item()),
        "seg/macro_precision_mean": float(precision.mean().item()),
        "seg/macro_recall_mean": float(recall.mean().item()),
        "seg/pred_foreground_ratio": float(pred.mean().item()),
        "seg/target_foreground_ratio": float(target.mean().item()),
    }

    channel_names = channel_names or [f"channel_{idx}" for idx in range(pred.shape[1])]
    for idx, name in enumerate(channel_names):
        metrics[f"seg/{name}/dice"] = float(dice[:, idx].mean().item())
        metrics[f"seg/{name}/iou"] = float(iou[:, idx].mean().item())
        metrics[f"seg/{name}/precision"] = float(precision[:, idx].mean().item())
        metrics[f"seg/{name}/recall"] = float(recall[:, idx].mean().item())
    return metrics


def brats_region_metrics(
    logits: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    channel_names: list[str] | None = None,
    region_channels: dict[str, list[int]] | None = None,
    eps: float = 1e-6,
) -> dict[str, float]:
    pred, target = _threshold_predictions(logits, target, threshold)
    metrics = multilabel_segmentation_metrics(logits, target, threshold, channel_names, eps)

    region_channels = region_channels or {"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]}
    region_dice: list[float] = []
    for region_name, channels in region_channels.items():
        region_pred = pred[:, channels].amax(dim=1, keepdim=True)
        region_target = target[:, channels].amax(dim=1, keepdim=True)
        dice, iou, precision, recall = _segmentation_stats(region_pred, region_target, eps)
        prefix = f"brats/{region_name}"
        metrics[f"{prefix}/dice"] = float(dice.mean().item())
        metrics[f"{prefix}/iou"] = float(iou.mean().item())
        metrics[f"{prefix}/precision"] = float(precision.mean().item())
        metrics[f"{prefix}/recall"] = float(recall.mean().item())
        metrics[f"{prefix}/pred_foreground_ratio"] = float(region_pred.mean().item())
        metrics[f"{prefix}/target_foreground_ratio"] = float(region_target.mean().item())
        region_dice.append(metrics[f"{prefix}/dice"])

    metrics["brats/mean_dice"] = float(np.mean(region_dice)) if region_dice else float("nan")
    return metrics


def multilabel_volume_metrics_from_probs(
    probs: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    channel_names: list[str] | None = None,
    voxel_spacing: tuple[float, ...] | None = None,
    eps: float = 1e-6,
) -> dict[str, float]:
    pred = (probs >= threshold).float()
    target = target.float()
    pred_volume = _volume_layout(pred)
    target_volume = _volume_layout(target)
    dice, iou, precision, recall = _segmentation_stats(pred_volume, target_volume, eps)
    pred_np = pred.detach().cpu().numpy() > 0.5
    target_np = target.detach().cpu().numpy() > 0.5

    metrics = {
        "seg/macro_dice_mean": float(dice.mean().item()),
        "seg/macro_iou_mean": float(iou.mean().item()),
        "seg/macro_precision_mean": float(precision.mean().item()),
        "seg/macro_recall_mean": float(recall.mean().item()),
        "seg/pred_foreground_ratio": float(pred.mean().item()),
        "seg/target_foreground_ratio": float(target.mean().item()),
    }
    channel_names = channel_names or [f"channel_{idx}" for idx in range(pred.shape[1])]
    hd95_values: list[float] = []
    for idx, name in enumerate(channel_names):
        metrics[f"seg/{name}/dice"] = float(dice[0, idx].item())
        metrics[f"seg/{name}/iou"] = float(iou[0, idx].item())
        metrics[f"seg/{name}/precision"] = float(precision[0, idx].item())
        metrics[f"seg/{name}/recall"] = float(recall[0, idx].item())
        hd95 = hd95_from_masks(pred_np[:, idx], target_np[:, idx], voxel_spacing)
        metrics[f"seg/{name}/hd95"] = hd95
        hd95_values.append(hd95)
    metrics["seg/macro_hd95_mean"] = float(np.mean(hd95_values)) if hd95_values else float("nan")
    return metrics


def brats_volume_metrics_from_probs(
    probs: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    channel_names: list[str] | None = None,
    region_channels: dict[str, list[int]] | None = None,
    voxel_spacing: tuple[float, ...] | None = None,
    eps: float = 1e-6,
) -> dict[str, float]:
    pred = (probs >= threshold).float()
    target = target.float()
    metrics = multilabel_volume_metrics_from_probs(probs, target, threshold, channel_names, voxel_spacing, eps)

    region_channels = region_channels or {"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]}
    region_dice: list[float] = []
    region_hd95: list[float] = []
    for region_name, channels in region_channels.items():
        region_pred = pred[:, channels].amax(dim=1, keepdim=True)
        region_target = target[:, channels].amax(dim=1, keepdim=True)
        dice, iou, precision, recall = _segmentation_stats(_volume_layout(region_pred), _volume_layout(region_target), eps)
        prefix = f"brats/{region_name}"
        metrics[f"{prefix}/dice"] = float(dice[0, 0].item())
        metrics[f"{prefix}/iou"] = float(iou[0, 0].item())
        metrics[f"{prefix}/precision"] = float(precision[0, 0].item())
        metrics[f"{prefix}/recall"] = float(recall[0, 0].item())
        metrics[f"{prefix}/pred_foreground_ratio"] = float(region_pred.mean().item())
        metrics[f"{prefix}/target_foreground_ratio"] = float(region_target.mean().item())
        region_dice.append(metrics[f"{prefix}/dice"])
        hd95 = hd95_from_masks(
            region_pred[:, 0].detach().cpu().numpy() > 0.5,
            region_target[:, 0].detach().cpu().numpy() > 0.5,
            voxel_spacing,
        )
        metrics[f"{prefix}/hd95"] = hd95
        region_hd95.append(hd95)
    metrics["brats/mean_dice"] = float(np.mean(region_dice)) if region_dice else float("nan")
    metrics["brats/mean_hd95"] = float(np.mean(region_hd95)) if region_hd95 else float("nan")
    return metrics


def classification_metrics(logits: Tensor, target: Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    target = target.float()
    if target.ndim == 1:
        target = target.unsqueeze(1)
    if probs.ndim == 1:
        probs = probs.unsqueeze(1)

    probs_np = _as_numpy(probs)
    target_np = _as_numpy(target)
    pred_np = (probs_np >= threshold).astype(np.int64)

    metrics: dict[str, float] = {
        "cls/prob_mean": float(probs_np.mean()),
        "cls/label_mean": float(target_np.mean()),
        "cls/accuracy": float(accuracy_score(target_np.reshape(-1), pred_np.reshape(-1))),
        "cls/balanced_accuracy": float(balanced_accuracy_score(target_np.reshape(-1), pred_np.reshape(-1))),
    }

    if probs_np.shape[1] == 1:
        y_true = target_np[:, 0]
        y_prob = probs_np[:, 0]
        y_pred = pred_np[:, 0]
        positives = y_true == 1
        negatives = y_true == 0
        metrics["cls/sensitivity"] = float((y_pred[positives] == 1).mean()) if positives.any() else float("nan")
        metrics["cls/specificity"] = float((y_pred[negatives] == 0).mean()) if negatives.any() else float("nan")
        if len(np.unique(y_true)) > 1:
            metrics["cls/auroc"] = float(roc_auc_score(y_true, y_prob))
            metrics["cls/average_precision"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["cls/auroc"] = float("nan")
            metrics["cls/average_precision"] = float("nan")
        return metrics

    aurocs: list[float] = []
    aps: list[float] = []
    for idx in range(probs_np.shape[1]):
        y_true = target_np[:, idx]
        y_prob = probs_np[:, idx]
        if len(np.unique(y_true)) > 1:
            aurocs.append(float(roc_auc_score(y_true, y_prob)))
            aps.append(float(average_precision_score(y_true, y_prob)))
    metrics["cls/macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    metrics["cls/macro_average_precision"] = float(np.mean(aps)) if aps else float("nan")
    return metrics


def summarize_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items]
        summary[key] = float(np.mean(values))
    return summary
