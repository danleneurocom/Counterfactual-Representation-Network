from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from typing import Any

import torch
from torch import Tensor


REGION_NAMES = ("WT", "TC", "ET")
CALIBRATION_OBJECTIVES = ("mean", "tc_et_mean", "tc_et_min", "et")


def prefix_metrics(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def parse_threshold_candidates(spec: str | None) -> list[float]:
    if spec is None or str(spec).strip() == "":
        return []
    values = [float(item.strip()) for item in str(spec).split(",") if item.strip()]
    thresholds = sorted({round(value, 4) for value in values})
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Calibration thresholds must be between 0 and 1, got {threshold}")
    return thresholds


def parse_fraction_candidates(spec: str | None) -> list[float]:
    if spec is None or str(spec).strip() == "":
        return []
    values = [float(item.strip()) for item in str(spec).split(",") if item.strip()]
    thresholds = sorted({round(value, 6) for value in values})
    for threshold in thresholds:
        if not 0.0 <= threshold < 1.0:
            raise ValueError(f"Fraction candidates must be in [0, 1), got {threshold}")
    return thresholds


def parse_region_thresholds(spec: str | None) -> dict[str, float] | None:
    if spec is None or str(spec).strip() == "":
        return None
    parts = [part.strip() for part in str(spec).split(",") if part.strip()]
    if len(parts) == 3 and all("=" not in part for part in parts):
        values = {name: float(value) for name, value in zip(REGION_NAMES, parts, strict=True)}
    else:
        values: dict[str, float] = {}
        for part in parts:
            if "=" not in part:
                raise ValueError(
                    "--region-thresholds must be 'WT,TC,ET' values or named values like 'WT=0.4,TC=0.5,ET=0.6'"
                )
            name, value = part.split("=", 1)
            key = name.strip().upper()
            if key not in REGION_NAMES:
                raise ValueError(f"Unknown BraTS region {name!r}; expected one of {REGION_NAMES}")
            values[key] = float(value.strip())
        missing = [name for name in REGION_NAMES if name not in values]
        if missing:
            raise ValueError(f"Missing region thresholds for: {', '.join(missing)}")
    for name, threshold in values.items():
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(f"{name} threshold must be between 0 and 1, got {threshold}")
    return {name: float(values[name]) for name in REGION_NAMES}


def _as_threshold_tensor(thresholds: Mapping[str, float], device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor([float(thresholds[name]) for name in REGION_NAMES], device=device, dtype=dtype).view(1, 3, 1, 1, 1)


def brats_region_probabilities(logits: Tensor) -> Tensor:
    if logits.ndim != 5 or logits.shape[1] < 3:
        raise ValueError(f"Expected logits shaped [B, 3, D, H, W], got {tuple(logits.shape)}")
    probs = torch.sigmoid(logits)
    wt = probs[:, [0, 1, 2]].amax(dim=1, keepdim=True)
    tc = probs[:, [0, 2]].amax(dim=1, keepdim=True)
    et = probs[:, 2:3]
    return torch.cat([wt, tc, et], dim=1)


def brats_region_targets(target: Tensor) -> Tensor:
    if target.ndim == 4:
        target = target.unsqueeze(1)
    if target.ndim != 5 or target.shape[1] < 3:
        raise ValueError(f"Expected target shaped [B, 3, D, H, W], got {tuple(target.shape)}")
    target = target.float()
    wt = target[:, [0, 1, 2]].amax(dim=1, keepdim=True)
    tc = target[:, [0, 2]].amax(dim=1, keepdim=True)
    et = target[:, 2:3]
    return torch.cat([wt, tc, et], dim=1)


def _enforce_region_hierarchy(pred: Tensor) -> Tensor:
    pred = pred.bool().clone()
    pred[:, 1] |= pred[:, 2]
    pred[:, 0] |= pred[:, 1]
    return pred.float()


def _region_metric_dict(pred: Tensor, target: Tensor, eps: float = 1e-6) -> dict[str, float]:
    dims = tuple(range(2, pred.ndim))
    target = target.float()
    intersection = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    metrics: dict[str, float] = {
        "brats/pred_foreground_ratio": float(pred.mean().item()),
        "brats/target_foreground_ratio": float(target.mean().item()),
    }
    dice_values: list[float] = []
    for idx, name in enumerate(REGION_NAMES):
        prefix = f"brats/{name}"
        metrics[f"{prefix}/dice"] = float(dice[:, idx].mean().item())
        metrics[f"{prefix}/iou"] = float(iou[:, idx].mean().item())
        metrics[f"{prefix}/precision"] = float(precision[:, idx].mean().item())
        metrics[f"{prefix}/recall"] = float(recall[:, idx].mean().item())
        metrics[f"{prefix}/pred_foreground_ratio"] = float(pred[:, idx].mean().item())
        metrics[f"{prefix}/target_foreground_ratio"] = float(target[:, idx].mean().item())
        dice_values.append(metrics[f"{prefix}/dice"])
    metrics["brats/mean_dice"] = float(sum(dice_values) / len(dice_values))
    return metrics


def brats_region_metrics_from_thresholds(
    logits: Tensor,
    target: Tensor,
    thresholds: Mapping[str, float],
    eps: float = 1e-6,
) -> dict[str, float]:
    region_probs = brats_region_probabilities(logits.detach().cpu())
    region_target = brats_region_targets(target.detach().cpu())
    threshold_tensor = _as_threshold_tensor(thresholds, region_probs.device, region_probs.dtype)
    pred = _enforce_region_hierarchy(region_probs >= threshold_tensor)
    return _region_metric_dict(pred, region_target, eps=eps)


def brats_region_metrics_from_adaptive_thresholds(
    logits: Tensor,
    target: Tensor,
    base_thresholds: Mapping[str, float],
    low_thresholds: Mapping[str, float],
    *,
    wt_ratio_threshold: float,
    eps: float = 1e-6,
) -> dict[str, float]:
    region_probs = brats_region_probabilities(logits.detach().cpu())
    region_target = brats_region_targets(target.detach().cpu())
    base_tensor = _as_threshold_tensor(base_thresholds, region_probs.device, region_probs.dtype)
    low_tensor = _as_threshold_tensor(low_thresholds, region_probs.device, region_probs.dtype)
    base_pred = _enforce_region_hierarchy(region_probs >= base_tensor)
    low_pred = _enforce_region_hierarchy(region_probs >= low_tensor)
    base_wt_ratio = base_pred[:, 0].float().flatten(1).mean(dim=1)
    use_low = base_wt_ratio < float(wt_ratio_threshold)
    pred = torch.where(use_low.view(-1, 1, 1, 1, 1), low_pred, base_pred)
    metrics = _region_metric_dict(pred, region_target, eps=eps)
    metrics["adaptive/low_threshold_fraction"] = float(use_low.float().mean().item())
    metrics["adaptive/low_threshold_count"] = float(use_low.float().sum().item())
    metrics["adaptive/base_WT_pred_foreground_ratio_mean"] = float(base_wt_ratio.mean().item())
    metrics["adaptive/base_WT_pred_foreground_ratio_min"] = float(base_wt_ratio.min().item())
    metrics["adaptive/base_WT_pred_foreground_ratio_max"] = float(base_wt_ratio.max().item())
    metrics["adaptive/wt_ratio_threshold"] = float(wt_ratio_threshold)
    for name in REGION_NAMES:
        metrics[f"adaptive/base_threshold/{name}"] = float(base_thresholds[name])
        metrics[f"adaptive/low_threshold/{name}"] = float(low_thresholds[name])
    return metrics


def brats_region_metrics_from_plausibility_thresholds(
    logits: Tensor,
    target: Tensor,
    base_thresholds: Mapping[str, float],
    low_thresholds: Mapping[str, float],
    *,
    low_stability_wt_ratio_threshold: float,
    low_stability_threshold: float,
    tc_collapse_wt_ratio_min: float,
    tc_collapse_wt_ratio_max: float,
    tc_collapse_tc_ratio_threshold: float,
    stability_scores: Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, float]:
    region_probs = brats_region_probabilities(logits.detach().cpu())
    region_target = brats_region_targets(target.detach().cpu())
    base_tensor = _as_threshold_tensor(base_thresholds, region_probs.device, region_probs.dtype)
    low_tensor = _as_threshold_tensor(low_thresholds, region_probs.device, region_probs.dtype)
    base_pred = _enforce_region_hierarchy(region_probs >= base_tensor)
    low_pred = _enforce_region_hierarchy(region_probs >= low_tensor)
    base_wt_ratio = base_pred[:, 0].float().flatten(1).mean(dim=1)
    base_tc_ratio = base_pred[:, 1].float().flatten(1).mean(dim=1)

    if stability_scores is None:
        stability = torch.ones_like(base_wt_ratio)
    else:
        stability = stability_scores.detach().cpu().to(dtype=base_wt_ratio.dtype).flatten()
        if stability.numel() != base_wt_ratio.numel():
            raise ValueError(
                "stability_scores must have one value per batch item, "
                f"got {stability.numel()} for batch {base_wt_ratio.numel()}"
            )

    low_stability_gate = (base_wt_ratio < float(low_stability_wt_ratio_threshold)) & (
        stability < float(low_stability_threshold)
    )
    tc_collapse_gate = (
        (base_wt_ratio >= float(tc_collapse_wt_ratio_min))
        & (base_wt_ratio < float(tc_collapse_wt_ratio_max))
        & (base_tc_ratio < float(tc_collapse_tc_ratio_threshold))
    )
    use_low = low_stability_gate | tc_collapse_gate
    pred = torch.where(use_low.view(-1, 1, 1, 1, 1), low_pred, base_pred)
    metrics = _region_metric_dict(pred, region_target, eps=eps)
    metrics["plausibility/low_threshold_fraction"] = float(use_low.float().mean().item())
    metrics["plausibility/low_threshold_count"] = float(use_low.float().sum().item())
    metrics["plausibility/low_stability_gate_fraction"] = float(low_stability_gate.float().mean().item())
    metrics["plausibility/tc_collapse_gate_fraction"] = float(tc_collapse_gate.float().mean().item())
    metrics["plausibility/base_WT_pred_foreground_ratio_mean"] = float(base_wt_ratio.mean().item())
    metrics["plausibility/base_WT_pred_foreground_ratio_min"] = float(base_wt_ratio.min().item())
    metrics["plausibility/base_WT_pred_foreground_ratio_max"] = float(base_wt_ratio.max().item())
    metrics["plausibility/base_TC_pred_foreground_ratio_mean"] = float(base_tc_ratio.mean().item())
    metrics["plausibility/stability_score_mean"] = float(stability.mean().item())
    metrics["plausibility/low_stability_wt_ratio_threshold"] = float(low_stability_wt_ratio_threshold)
    metrics["plausibility/low_stability_threshold"] = float(low_stability_threshold)
    metrics["plausibility/tc_collapse_wt_ratio_min"] = float(tc_collapse_wt_ratio_min)
    metrics["plausibility/tc_collapse_wt_ratio_max"] = float(tc_collapse_wt_ratio_max)
    metrics["plausibility/tc_collapse_tc_ratio_threshold"] = float(tc_collapse_tc_ratio_threshold)
    for name in REGION_NAMES:
        metrics[f"plausibility/base_threshold/{name}"] = float(base_thresholds[name])
        metrics[f"plausibility/low_threshold/{name}"] = float(low_thresholds[name])
    return metrics


def _record_metric_float(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    if value is None:
        raise ValueError(f"Missing required plausibility support metric {key!r}")
    return float(value)


def fit_plausibility_support_thresholds(
    support_records: Sequence[Mapping[str, Any]],
    *,
    validation_records: Sequence[Mapping[str, Any]] | None = None,
    prefix: str = "registered_tta_plausibility_region_calibrated",
    low_stability_threshold: float = 0.90,
    low_stability_wt_margin: float = 0.95,
    tc_collapse_tc_margin: float = 0.50,
    tc_collapse_wt_ratio_min: float = 0.0,
) -> dict[str, float]:
    """Fit H76 plausibility gates from train/validation support records.

    The fit is label-free: it only uses prediction ratios and registered-view
    stability emitted by per-case evaluation. Margins keep the fitted gates just
    outside observed support so the rule is conservative on the fitting cases.
    """

    if not support_records:
        raise ValueError("At least one support record is required.")
    if not 0.0 < low_stability_wt_margin <= 1.0:
        raise ValueError("low_stability_wt_margin must be in (0, 1].")
    if not 0.0 < tc_collapse_tc_margin <= 1.0:
        raise ValueError("tc_collapse_tc_margin must be in (0, 1].")

    wt_key = f"{prefix}/plausibility/base_WT_pred_foreground_ratio_mean"
    tc_key = f"{prefix}/plausibility/base_TC_pred_foreground_ratio_mean"
    stability_key = f"{prefix}/plausibility/stability_score_mean"

    wt_values = [_record_metric_float(record, wt_key) for record in support_records]
    tc_values = [_record_metric_float(record, tc_key) for record in support_records]
    stability_values = [_record_metric_float(record, stability_key) for record in support_records]

    low_stability_wt_values = [
        wt
        for wt, stability in zip(wt_values, stability_values, strict=True)
        if stability < float(low_stability_threshold)
    ]
    if low_stability_wt_values:
        low_stability_wt_floor = min(low_stability_wt_values)
    else:
        low_stability_wt_floor = min(wt_values)

    validation_source = validation_records if validation_records is not None else support_records
    if not validation_source:
        raise ValueError("validation_records must not be empty when supplied.")
    validation_wt_values = [_record_metric_float(record, wt_key) for record in validation_source]

    min_tc_ratio = min(tc_values)
    min_validation_wt_ratio = min(validation_wt_values)
    out = {
        "plausibility/low_stability_wt_ratio_threshold": float(low_stability_wt_floor * low_stability_wt_margin),
        "plausibility/low_stability_threshold": float(low_stability_threshold),
        "plausibility/tc_collapse_wt_ratio_min": float(tc_collapse_wt_ratio_min),
        "plausibility/tc_collapse_wt_ratio_max": float(min_validation_wt_ratio),
        "plausibility/tc_collapse_tc_ratio_threshold": float(min_tc_ratio * tc_collapse_tc_margin),
        "plausibility/support_case_count": float(len(support_records)),
        "plausibility/validation_case_count": float(len(validation_source)),
        "plausibility/support_min_WT_pred_foreground_ratio": float(min(wt_values)),
        "plausibility/support_min_TC_pred_foreground_ratio": float(min_tc_ratio),
        "plausibility/support_min_stability_score": float(min(stability_values)),
        "plausibility/support_low_stability_case_count": float(len(low_stability_wt_values)),
        "plausibility/support_low_stability_min_WT_pred_foreground_ratio": float(low_stability_wt_floor),
        "plausibility/validation_min_WT_pred_foreground_ratio": float(min_validation_wt_ratio),
    }
    return out


def _mean_dicts(items: Iterable[Mapping[str, float]]) -> dict[str, float]:
    item_list = list(items)
    if not item_list:
        return {}
    keys = sorted({key for item in item_list for key in item})
    return {
        key: float(sum(float(item[key]) for item in item_list if key in item) / max(sum(1 for item in item_list if key in item), 1))
        for key in keys
    }


class BratsRegionThresholdSweep:
    """Grid-search WT/TC/ET thresholds while preserving ET <= TC <= WT."""

    def __init__(self, candidates: Sequence[float], objective: str = "mean") -> None:
        thresholds = sorted({round(float(value), 4) for value in candidates})
        if not thresholds:
            raise ValueError("At least one calibration threshold is required.")
        objective = str(objective).strip().lower().replace("-", "_")
        if objective not in CALIBRATION_OBJECTIVES:
            valid = ", ".join(CALIBRATION_OBJECTIVES)
            raise ValueError(f"Unknown calibration objective {objective!r}; expected one of: {valid}")
        self.candidates = thresholds
        self.objective = objective
        self._items: dict[tuple[float, float, float], list[dict[str, float]]] = {
            combo: [] for combo in product(thresholds, repeat=3)
        }

    def update(self, logits: Tensor, target: Tensor) -> None:
        region_probs = brats_region_probabilities(logits.detach().cpu())
        region_target = brats_region_targets(target.detach().cpu())
        for combo, items in self._items.items():
            thresholds = {name: combo[idx] for idx, name in enumerate(REGION_NAMES)}
            threshold_tensor = _as_threshold_tensor(thresholds, region_probs.device, region_probs.dtype)
            pred = _enforce_region_hierarchy(region_probs >= threshold_tensor)
            items.append(_region_metric_dict(pred, region_target))

    def _score(self, metrics: Mapping[str, float]) -> tuple[float, ...]:
        wt = float(metrics.get("brats/WT/dice", float("-inf")))
        tc = float(metrics.get("brats/TC/dice", float("-inf")))
        et = float(metrics.get("brats/ET/dice", float("-inf")))
        mean = float(metrics.get("brats/mean_dice", float("-inf")))
        if self.objective == "mean":
            return (mean,)
        if self.objective == "tc_et_mean":
            return ((tc + et) / 2.0, mean, wt)
        if self.objective == "tc_et_min":
            return (min(tc, et), (tc + et) / 2.0, mean, wt)
        if self.objective == "et":
            return (et, min(tc, et), mean, wt)
        raise AssertionError(f"Unhandled calibration objective: {self.objective}")

    def summary(self) -> dict[str, float]:
        best_combo: tuple[float, float, float] | None = None
        best_metrics: dict[str, float] | None = None
        best_score: tuple[float, ...] | None = None
        for combo, items in self._items.items():
            metrics = _mean_dicts(items)
            score = self._score(metrics)
            if best_score is None or score > best_score:
                best_combo = combo
                best_metrics = metrics
                best_score = score
        if best_combo is None or best_metrics is None:
            return {}
        out: dict[str, float] = dict(best_metrics)
        for idx, name in enumerate(REGION_NAMES):
            out[f"threshold/{name}"] = float(best_combo[idx])
        out["threshold/grid_size"] = float(len(self._items))
        out["threshold/objective_score"] = float(best_score[0])
        return out


class BratsAdaptiveRegionThresholdSweep:
    """Sweep low-confidence case calibration while keeping base thresholds fixed."""

    def __init__(
        self,
        base_thresholds: Mapping[str, float],
        low_candidates: Sequence[float],
        wt_ratio_candidates: Sequence[float],
    ) -> None:
        low_values = sorted({round(float(value), 4) for value in low_candidates})
        ratio_values = sorted({round(float(value), 6) for value in wt_ratio_candidates})
        if not low_values:
            raise ValueError("At least one low-threshold candidate is required.")
        if not ratio_values:
            raise ValueError("At least one WT-ratio candidate is required.")
        self.base_thresholds = {name: float(base_thresholds[name]) for name in REGION_NAMES}
        self.low_candidates = low_values
        self.wt_ratio_candidates = ratio_values
        self._items: dict[tuple[float, float, float, float], list[dict[str, float]]] = {
            (*combo, ratio): [] for combo in product(low_values, repeat=3) for ratio in ratio_values
        }

    def update(self, logits: Tensor, target: Tensor) -> None:
        region_probs = brats_region_probabilities(logits.detach().cpu())
        region_target = brats_region_targets(target.detach().cpu())
        base_tensor = _as_threshold_tensor(self.base_thresholds, region_probs.device, region_probs.dtype)
        base_pred = _enforce_region_hierarchy(region_probs >= base_tensor)
        base_wt_ratio = base_pred[:, 0].float().flatten(1).mean(dim=1)
        low_pred_cache: dict[tuple[float, float, float], Tensor] = {}
        for key, items in self._items.items():
            low_key = key[:3]
            ratio = key[3]
            low_pred = low_pred_cache.get(low_key)
            if low_pred is None:
                low_thresholds = {name: low_key[idx] for idx, name in enumerate(REGION_NAMES)}
                low_tensor = _as_threshold_tensor(low_thresholds, region_probs.device, region_probs.dtype)
                low_pred = _enforce_region_hierarchy(region_probs >= low_tensor)
                low_pred_cache[low_key] = low_pred
            use_low = base_wt_ratio < ratio
            pred = torch.where(use_low.view(-1, 1, 1, 1, 1), low_pred, base_pred)
            metrics = _region_metric_dict(pred, region_target)
            metrics["adaptive/low_threshold_fraction"] = float(use_low.float().mean().item())
            metrics["adaptive/low_threshold_count"] = float(use_low.float().sum().item())
            metrics["adaptive/base_WT_pred_foreground_ratio_mean"] = float(base_wt_ratio.mean().item())
            metrics["adaptive/base_WT_pred_foreground_ratio_min"] = float(base_wt_ratio.min().item())
            metrics["adaptive/base_WT_pred_foreground_ratio_max"] = float(base_wt_ratio.max().item())
            items.append(metrics)

    def summary(self) -> dict[str, float]:
        best_key: tuple[float, float, float, float] | None = None
        best_metrics: dict[str, float] | None = None
        best_rank: tuple[float, float, float, float] | None = None
        for key, items in self._items.items():
            metrics = _mean_dicts(items)
            score = float(metrics.get("brats/mean_dice", float("-inf")))
            low_fraction = float(metrics.get("adaptive/low_threshold_fraction", 0.0))
            ratio = float(key[3])
            low_sum = float(sum(key[:3]))
            rank = (score, -low_fraction, -ratio, low_sum)
            if best_rank is None or rank > best_rank:
                best_key = key
                best_metrics = metrics
                best_rank = rank
        if best_key is None or best_metrics is None:
            return {}
        out: dict[str, float] = dict(best_metrics)
        for idx, name in enumerate(REGION_NAMES):
            out[f"adaptive/base_threshold/{name}"] = float(self.base_thresholds[name])
            out[f"adaptive/low_threshold/{name}"] = float(best_key[idx])
        out["adaptive/wt_ratio_threshold"] = float(best_key[3])
        out["adaptive/grid_size"] = float(len(self._items))
        return out


__all__ = [
    "BratsAdaptiveRegionThresholdSweep",
    "BratsRegionThresholdSweep",
    "CALIBRATION_OBJECTIVES",
    "brats_region_metrics_from_adaptive_thresholds",
    "brats_region_metrics_from_plausibility_thresholds",
    "brats_region_metrics_from_thresholds",
    "fit_plausibility_support_thresholds",
    "parse_fraction_candidates",
    "parse_region_thresholds",
    "parse_threshold_candidates",
    "prefix_metrics",
]
