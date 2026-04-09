from __future__ import annotations

from typing import Any

import numpy as np
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
