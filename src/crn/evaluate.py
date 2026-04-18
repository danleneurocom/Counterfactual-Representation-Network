from __future__ import annotations

import argparse
from itertools import islice
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm

from crn.data import ImagingCsvDataset, make_dataloader
from crn.losses import backdoor_adjusted_seg_logits, compute_crn_losses
from crn.metrics import (
    binary_segmentation_metrics,
    binary_volume_metrics_from_masks,
    brats_region_arrays_from_volume,
    brats_region_metrics,
    brats_volume_metrics_from_probs,
    classification_metrics,
    multilabel_segmentation_metrics,
    multilabel_volume_metrics_from_probs,
    postprocess_binary_volume,
)
from crn.train import build_model
from crn.utils import load_yaml, move_batch_to_device, resolve_device, save_json


def _load_config(checkpoint: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    if config_path:
        return load_yaml(config_path)
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain a saved config; pass --config explicitly.")
    return checkpoint["config"]


def _weighted_average_dicts(items: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    if not items:
        return {}
    totals = {key: 0.0 for key in items[0]}
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("Weights must sum to a positive value.")
    for item, weight in zip(items, weights, strict=False):
        for key, value in item.items():
            totals[key] += float(value) * float(weight)
    return {key: value / weight_sum for key, value in totals.items()}


def _parse_threshold_sweep(spec: str | None) -> list[float]:
    if not spec:
        return []
    values = [chunk.strip() for chunk in spec.split(",") if chunk.strip()]
    thresholds = sorted({round(float(value), 4) for value in values})
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold sweep values must be between 0 and 1, got {threshold}")
    return thresholds


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, data_config: dict[str, Any], threshold: float) -> dict[str, float]:
    mask_mode = data_config.get("mask_mode")
    if target.ndim == 3:
        return binary_segmentation_metrics(logits, target, threshold)
    if target.ndim == 4 and target.shape[1] == 1:
        return binary_segmentation_metrics(logits, target, threshold)
    if mask_mode == "brats_subregions":
        return brats_region_metrics(
            logits,
            target,
            threshold=threshold,
            channel_names=data_config.get("brats_channel_names"),
            region_channels=data_config.get("brats_region_channels"),
        )
    return multilabel_segmentation_metrics(logits, target, threshold, data_config.get("segmentation_channel_names"))


def _volume_segmentation_metrics(probs: torch.Tensor, target: torch.Tensor, data_config: dict[str, Any], threshold: float) -> dict[str, float]:
    mask_mode = data_config.get("mask_mode")
    voxel_spacing = tuple(float(value) for value in data_config.get("voxel_spacing", (1.0, 1.0, 1.0)))
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim == 4 and target.shape[1] == 1:
        return multilabel_volume_metrics_from_probs(probs, target, threshold, ["foreground"], voxel_spacing=voxel_spacing)
    if mask_mode == "brats_subregions":
        return brats_volume_metrics_from_probs(
            probs,
            target,
            threshold=threshold,
            channel_names=data_config.get("brats_channel_names"),
            region_channels=data_config.get("brats_region_channels"),
            voxel_spacing=voxel_spacing,
        )
    return multilabel_volume_metrics_from_probs(
        probs,
        target,
        threshold,
        data_config.get("segmentation_channel_names"),
        voxel_spacing=voxel_spacing,
    )


def _threshold_score(metrics: dict[str, float]) -> float:
    region_keys = [key for key in ("brats/WT/dice", "brats/TC/dice", "brats/ET/dice") if key in metrics]
    if region_keys:
        values = [metrics[key] for key in region_keys if math.isfinite(metrics[key])]
        return float(sum(values) / len(values)) if values else float("-inf")
    if "seg/macro_dice_mean" in metrics and math.isfinite(metrics["seg/macro_dice_mean"]):
        return float(metrics["seg/macro_dice_mean"])
    return float("-inf")


def _region_candidate_key(threshold: float, config: dict[str, Any]) -> str:
    return (
        f"thr={threshold:.4f}|fill={int(bool(config.get('fill_holes', False)))}"
        f"|min={int(config.get('min_component_size', 0))}"
        f"|largest={int(bool(config.get('keep_largest', False)))}"
    )


def _region_candidate_sort_value(metrics: dict[str, float]) -> tuple[float, float, float]:
    dice = metrics.get("dice", float("-inf"))
    hd95 = metrics.get("hd95", float("inf"))
    recall = metrics.get("recall", float("-inf"))
    dice = dice if math.isfinite(dice) else float("-inf")
    hd95 = hd95 if math.isfinite(hd95) else float("inf")
    recall = recall if math.isfinite(recall) else float("-inf")
    return (dice, -hd95, recall)


def _default_brats_region_postprocess_candidates() -> dict[str, list[dict[str, Any]]]:
    return {
        "WT": [
            {"name": "raw", "min_component_size": 0, "fill_holes": False, "keep_largest": False},
            {"name": "clean", "min_component_size": 256, "fill_holes": True, "keep_largest": False},
        ],
        "TC": [
            {"name": "raw", "min_component_size": 0, "fill_holes": False, "keep_largest": False},
            {"name": "clean", "min_component_size": 96, "fill_holes": True, "keep_largest": False},
        ],
        "ET": [
            {"name": "raw", "min_component_size": 0, "fill_holes": False, "keep_largest": False},
            {"name": "clean", "min_component_size": 32, "fill_holes": True, "keep_largest": False},
        ],
    }


def _resolve_brats_region_postprocess_candidates(data_config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    region_channels = data_config.get("brats_region_channels") or {"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]}
    defaults = _default_brats_region_postprocess_candidates()
    configured = data_config.get("brats_region_postprocessing")
    if not configured:
        return {
            region_name: [dict(candidate) for candidate in defaults.get(region_name, [{"min_component_size": 0, "fill_holes": False}])]
            for region_name in region_channels
        }

    resolved: dict[str, list[dict[str, Any]]] = {}
    for region_name in region_channels:
        candidates = configured.get(region_name) if isinstance(configured, dict) else None
        if not candidates:
            candidates = defaults.get(region_name, [{"min_component_size": 0, "fill_holes": False}])
        resolved[region_name] = [dict(candidate) for candidate in candidates]
    return resolved


class VolumeMetricTracker:
    def __init__(
        self,
        thresholds: list[float],
        data_config: dict[str, Any],
        *,
        region_thresholds: list[float] | None = None,
        region_postprocess_candidates: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.data_config = data_config
        self.summary_by_threshold: dict[float, list[dict[str, float]]] = {threshold: [] for threshold in thresholds}
        self.per_volume_by_threshold: dict[float, dict[int, dict[str, float]]] = {threshold: {} for threshold in thresholds}
        self.current_volume: int | None = None
        self.current_slices: list[int] = []
        self.current_probs: list[torch.Tensor] = []
        self.current_targets: list[torch.Tensor] = []
        self.seen_volumes: set[int] = set()
        self.voxel_spacing = tuple(float(value) for value in data_config.get("voxel_spacing", (1.0, 1.0, 1.0)))

        self.region_thresholds = region_thresholds or []
        self.region_channels = data_config.get("brats_region_channels") or {"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]}
        self.region_postprocess_candidates = region_postprocess_candidates or {}
        self.region_candidate_summaries: dict[str, dict[str, list[dict[str, float]]]] = {
            region_name: {} for region_name in self.region_channels
        }
        self.region_candidate_per_volume: dict[str, dict[str, dict[int, dict[str, float]]]] = {
            region_name: {} for region_name in self.region_channels
        }
        self.region_candidate_configs: dict[str, dict[str, dict[str, Any]]] = {
            region_name: {} for region_name in self.region_channels
        }

    def add_batch(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        volumes: torch.Tensor | None,
        slices: torch.Tensor | None,
    ) -> None:
        if volumes is None or slices is None:
            return
        probs = torch.sigmoid(logits.detach()).cpu().to(torch.float16)
        targets = target.detach().cpu().to(torch.uint8)
        volume_ids = volumes.detach().cpu().tolist()
        slice_ids = slices.detach().cpu().tolist()
        for index, volume_id in enumerate(volume_ids):
            self._add_sample(probs[index], targets[index], int(volume_id), int(slice_ids[index]))

    def finalize(
        self,
    ) -> tuple[
        dict[float, dict[str, float]],
        dict[float, dict[int, dict[str, float]]],
        dict[str, Any] | None,
    ]:
        self._flush_current()
        summaries = {
            threshold: _weighted_average_dicts(items, [1.0] * len(items)) if items else {}
            for threshold, items in self.summary_by_threshold.items()
        }
        region_tuning: dict[str, Any] | None = None
        if self.region_thresholds:
            region_tuning = self._finalize_region_tuning()
        return summaries, self.per_volume_by_threshold, region_tuning

    def _add_sample(self, probs: torch.Tensor, target: torch.Tensor, volume_id: int, slice_id: int) -> None:
        if self.current_volume is None:
            self._start_volume(volume_id)
        elif volume_id != self.current_volume:
            self._flush_current()
            if volume_id in self.seen_volumes:
                raise ValueError(
                    "Volume IDs are not grouped in evaluation order. Ensure the evaluation CSV is sorted by volume and slice."
                )
            self._start_volume(volume_id)

        self.current_probs.append(probs)
        self.current_targets.append(target)
        self.current_slices.append(slice_id)

    def _start_volume(self, volume_id: int) -> None:
        self.current_volume = volume_id
        self.current_slices = []
        self.current_probs = []
        self.current_targets = []

    def _flush_current(self) -> None:
        if self.current_volume is None:
            return
        order = sorted(range(len(self.current_slices)), key=self.current_slices.__getitem__)
        probs = torch.stack([self.current_probs[index] for index in order], dim=0).float()
        target = torch.stack([self.current_targets[index] for index in order], dim=0).float()
        for threshold in self.thresholds:
            metrics = _volume_segmentation_metrics(probs, target, self.data_config, threshold)
            self.summary_by_threshold[threshold].append(metrics)
            self.per_volume_by_threshold[threshold][self.current_volume] = metrics
        if self.region_thresholds:
            self._accumulate_region_tuning(probs, target, self.current_volume)
        self.seen_volumes.add(self.current_volume)
        self.current_volume = None
        self.current_slices = []
        self.current_probs = []
        self.current_targets = []

    def _accumulate_region_tuning(self, probs: torch.Tensor, target: torch.Tensor, volume_id: int) -> None:
        region_probs = brats_region_arrays_from_volume(probs, self.region_channels)
        region_targets = {
            region_name: masks >= 0.5
            for region_name, masks in brats_region_arrays_from_volume(target, self.region_channels).items()
        }

        for region_name in self.region_channels:
            target_mask = region_targets[region_name]
            candidates = self.region_postprocess_candidates.get(
                region_name,
                [{"name": "raw", "min_component_size": 0, "fill_holes": False, "keep_largest": False}],
            )
            for threshold in self.region_thresholds:
                base_mask = region_probs[region_name] >= threshold
                for candidate in candidates:
                    candidate_key = _region_candidate_key(threshold, candidate)
                    pred_mask = postprocess_binary_volume(
                        base_mask,
                        min_component_size=int(candidate.get("min_component_size", 0)),
                        fill_holes=bool(candidate.get("fill_holes", False)),
                        keep_largest=bool(candidate.get("keep_largest", False)),
                    )
                    metrics = binary_volume_metrics_from_masks(pred_mask, target_mask, voxel_spacing=self.voxel_spacing)
                    self.region_candidate_summaries[region_name].setdefault(candidate_key, []).append(metrics)
                    self.region_candidate_per_volume[region_name].setdefault(candidate_key, {})[volume_id] = metrics
                    self.region_candidate_configs[region_name][candidate_key] = {
                        "threshold": float(threshold),
                        "min_component_size": int(candidate.get("min_component_size", 0)),
                        "fill_holes": bool(candidate.get("fill_holes", False)),
                        "keep_largest": bool(candidate.get("keep_largest", False)),
                        "name": candidate.get("name", "candidate"),
                    }

    def _finalize_region_tuning(self) -> dict[str, Any]:
        selected_configs: dict[str, dict[str, Any]] = {}
        selected_metrics: dict[str, dict[str, float]] = {}
        candidate_results: dict[str, dict[str, dict[str, Any]]] = {}
        per_volume_flat: dict[int, dict[str, float]] = {}
        mean_dice_values: list[float] = []
        mean_hd95_values: list[float] = []

        for region_name, candidates in self.region_candidate_summaries.items():
            if not candidates:
                continue
            region_candidate_results: dict[str, dict[str, Any]] = {}
            for candidate_key, items in candidates.items():
                summary = _weighted_average_dicts(items, [1.0] * len(items))
                region_candidate_results[candidate_key] = {
                    "config": self.region_candidate_configs[region_name][candidate_key],
                    "metrics": summary,
                }
            best_candidate_key = max(
                region_candidate_results,
                key=lambda key: _region_candidate_sort_value(region_candidate_results[key]["metrics"]),
            )
            selected_configs[region_name] = dict(region_candidate_results[best_candidate_key]["config"])
            selected_metrics[region_name] = dict(region_candidate_results[best_candidate_key]["metrics"])
            candidate_results[region_name] = region_candidate_results

            selected_per_volume = self.region_candidate_per_volume[region_name][best_candidate_key]
            for volume_id, metrics in selected_per_volume.items():
                volume_summary = per_volume_flat.setdefault(volume_id, {})
                for metric_name, value in metrics.items():
                    volume_summary[f"brats/{region_name}/{metric_name}"] = float(value)

        for volume_metrics in per_volume_flat.values():
            region_dice_values: list[float] = []
            region_hd95_values: list[float] = []
            for region_name in selected_metrics:
                dice_key = f"brats/{region_name}/dice"
                hd95_key = f"brats/{region_name}/hd95"
                if dice_key in volume_metrics:
                    region_dice_values.append(volume_metrics[dice_key])
                if hd95_key in volume_metrics:
                    region_hd95_values.append(volume_metrics[hd95_key])
            if region_dice_values:
                volume_metrics["brats/mean_dice"] = float(np.mean(region_dice_values))
            if region_hd95_values:
                volume_metrics["brats/mean_hd95"] = float(np.mean(region_hd95_values))

        flat_summary: dict[str, float] = {}
        for region_name, metrics in selected_metrics.items():
            for metric_name, value in metrics.items():
                flat_summary[f"brats/{region_name}/{metric_name}"] = float(value)
            if "dice" in metrics:
                mean_dice_values.append(metrics["dice"])
            if "hd95" in metrics:
                mean_hd95_values.append(metrics["hd95"])
        if mean_dice_values:
            flat_summary["brats/mean_dice"] = float(np.mean(mean_dice_values))
        if mean_hd95_values:
            flat_summary["brats/mean_hd95"] = float(np.mean(mean_hd95_values))

        return {
            "selected_configs": selected_configs,
            "selected_metrics": selected_metrics,
            "summary": flat_summary,
            "candidates": candidate_results,
            "per_volume": per_volume_flat,
        }


def _blank_panel(panel_size: int, label: str) -> Image.Image:
    image = Image.new("RGB", (panel_size, panel_size), color=(20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), label, fill=(220, 220, 220))
    return image


def _project_image_channels(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image.float()
    if image.ndim == 4:
        return _project_image_channels(image[:, image.shape[1] // 2])
    if image.shape[0] == 1:
        return image[0].float()
    return image.float().mean(dim=0)


def _channel_panel(channel: torch.Tensor, panel_size: int, label: str) -> Image.Image:
    array = channel.detach().cpu().numpy().astype(np.float32)
    array = array - array.min()
    scale = float(array.max())
    if scale > 0:
        array = array / scale
    image = Image.fromarray(np.uint8(np.clip(array * 255.0, 0, 255)), mode="L").resize(
        (panel_size, panel_size), Image.Resampling.BILINEAR
    )
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, panel_size - 1, 18), fill=(0, 0, 0))
    draw.text((6, 3), label, fill=(255, 255, 255))
    return image


def _heatmap_panel(
    base_channel: torch.Tensor,
    heatmap: torch.Tensor,
    panel_size: int,
    label: str,
    color: tuple[int, int, int] = (255, 120, 40),
) -> Image.Image:
    base = _channel_panel(base_channel, panel_size, label)
    base_array = np.asarray(base).astype(np.float32)
    heat = heatmap.detach().cpu().numpy().astype(np.float32)
    heat = np.clip(heat, a_min=0.0, a_max=None)
    max_value = float(heat.max())
    if max_value > 0.0:
        heat = heat / max_value
    heat_image = Image.fromarray(np.uint8(np.clip(heat * 255.0, 0, 255)), mode="L").resize(
        (panel_size, panel_size),
        Image.Resampling.BILINEAR,
    )
    heat_array = np.asarray(heat_image).astype(np.float32) / 255.0
    overlay = np.array(color, dtype=np.float32)
    alpha = 0.75 * heat_array[..., None]
    blended = (1.0 - alpha) * base_array + alpha * overlay
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


def _overlay_panel(base_channel: torch.Tensor, mask: torch.Tensor, panel_size: int, label: str, color: tuple[int, int, int]) -> Image.Image:
    base = _channel_panel(base_channel, panel_size, label)
    base_array = np.asarray(base).astype(np.float32)
    mask_array = mask.detach().cpu().numpy() > 0.5
    if mask_array.any():
        mask_image = Image.fromarray(np.uint8(mask_array) * 255, mode="L").resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        )
        mask_array = np.asarray(mask_image) > 0
        overlay = np.array(color, dtype=np.float32)
        base_array[mask_array] = 0.45 * base_array[mask_array] + 0.55 * overlay
    return Image.fromarray(np.uint8(np.clip(base_array, 0, 255)))


def _composite_overlay_panel(
    base_channel: torch.Tensor,
    masks: dict[str, torch.Tensor],
    panel_size: int,
    label: str,
    colors: dict[str, tuple[int, int, int]],
) -> Image.Image:
    image = _channel_panel(base_channel, panel_size, label)
    image_array = np.asarray(image).astype(np.float32)
    for name, mask in masks.items():
        mask_array = mask.detach().cpu().numpy() > 0.5
        if mask_array.any():
            mask_image = Image.fromarray(np.uint8(mask_array) * 255, mode="L").resize(
                (panel_size, panel_size), Image.Resampling.NEAREST
            )
            mask_array = np.asarray(mask_image) > 0
            image_array[mask_array] = 0.5 * image_array[mask_array] + 0.5 * np.array(colors[name], dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))


def _region_masks(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    if mask.ndim == 2:
        return {"WT": mask.float()}
    if mask.shape[0] == 1:
        return {"WT": mask[0].float()}
    regions = {
        "NCR": mask[0].float(),
        "ED": mask[1].float(),
        "ET": mask[2].float(),
    }
    regions["WT"] = mask.amax(dim=0).float()
    regions["TC"] = mask[[0, 2]].amax(dim=0).float()
    return regions


def _region_prob_maps(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    if mask.ndim == 2:
        return {"WT": mask.float().clamp(0.0, 1.0)}
    if mask.shape[0] == 1:
        return {"WT": mask[0].float().clamp(0.0, 1.0)}
    probs = mask.float().clamp(0.0, 1.0)
    regions = {
        "NCR": probs[0],
        "ED": probs[1],
        "ET": probs[2],
    }
    regions["WT"] = probs.amax(dim=0)
    regions["TC"] = probs[[0, 2]].amax(dim=0)
    return regions


def _region_effect_map(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    before_regions = _region_prob_maps(before)
    after_regions = _region_prob_maps(after)
    keys = [key for key in ("WT", "TC", "ET") if key in before_regions and key in after_regions]
    if not keys:
        shape = before.shape[-2:] if before.ndim >= 2 else after.shape[-2:]
        return torch.zeros(shape, dtype=torch.float32)
    deltas = [(after_regions[key] - before_regions[key]).abs() for key in keys]
    return torch.stack(deltas, dim=0).mean(dim=0)


def _cosine_distance(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    query = F.normalize(query.unsqueeze(0), dim=-1, eps=1e-6)
    bank = F.normalize(bank, dim=-1, eps=1e-6)
    return 1.0 - torch.sum(query * bank, dim=-1)


def _build_counterfactual_reference_bank(
    model: torch.nn.Module,
    dataset: ImagingCsvDataset,
    device: torch.device,
) -> list[dict[str, Any]]:
    if "volume" not in dataset.frame.columns:
        return []
    entries: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for volume_id in dataset.frame["volume"].drop_duplicates().tolist():
            row = _representative_row(dataset.frame, int(volume_id))
            row_index = int(row.name)
            sample = dataset[row_index]
            outputs = model(sample["image"].unsqueeze(0).to(device))
            entries.append(
                {
                    "volume": int(volume_id),
                    "row_index": row_index,
                    "slice": int(sample.get("slice", row.get("slice", -1))),
                    "path": sample["path"],
                    "z_d": outputs["z_d"][0].detach().cpu(),
                    "z_c": outputs["z_c"][0].detach().cpu(),
                }
            )
    return entries


def _select_counterfactual_references(
    reference_bank: list[dict[str, Any]],
    z_d: torch.Tensor,
    z_c: torch.Tensor,
    current_volume: int,
    topk: int = 8,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidates = [entry for entry in reference_bank if int(entry["volume"]) != int(current_volume)]
    if not candidates:
        candidates = list(reference_bank)
    if not candidates:
        return None, None
    candidate_z_d = torch.stack([entry["z_d"] for entry in candidates], dim=0).to(z_d.device)
    candidate_z_c = torch.stack([entry["z_c"] for entry in candidates], dim=0).to(z_c.device)
    disease_dist = _cosine_distance(z_d, candidate_z_d)
    context_dist = _cosine_distance(z_c, candidate_z_c)

    disease_topk = min(max(1, topk), int(disease_dist.numel()))
    disease_nearest = torch.topk(disease_dist, k=disease_topk, largest=False).indices
    context_entry = candidates[int(disease_nearest[torch.argmax(context_dist[disease_nearest])].item())]

    context_topk = min(max(1, topk), int(context_dist.numel()))
    context_nearest = torch.topk(context_dist, k=context_topk, largest=False).indices
    disease_entry = candidates[int(context_nearest[torch.argmax(disease_dist[context_nearest])].item())]
    return context_entry, disease_entry


def _tuned_region_masks(
    probs: torch.Tensor,
    region_channels: dict[str, list[int]],
    configs: dict[str, dict[str, Any]] | None,
) -> dict[str, torch.Tensor]:
    if not configs:
        return {}
    region_probs = brats_region_arrays_from_volume(probs.unsqueeze(0), region_channels)
    tuned: dict[str, torch.Tensor] = {}
    for region_name, config in configs.items():
        pred_mask = postprocess_binary_volume(
            region_probs[region_name] >= float(config["threshold"]),
            min_component_size=int(config.get("min_component_size", 0)),
            fill_holes=bool(config.get("fill_holes", False)),
            keep_largest=bool(config.get("keep_largest", False)),
        )
        tuned[region_name] = torch.from_numpy(pred_mask[0].astype(np.float32))
    return tuned


def _pick_qualitative_volumes(volume_metrics: dict[int, dict[str, float]], count: int) -> list[tuple[str, int, dict[str, float]]]:
    if count <= 0 or not volume_metrics:
        return []
    score_key = "brats/WT/dice" if any("brats/WT/dice" in metrics for metrics in volume_metrics.values()) else "seg/macro_dice_mean"
    ranked = sorted(volume_metrics.items(), key=lambda item: item[1].get(score_key, float("-inf")))
    take_each_side = max(1, count // 2)
    chosen: list[tuple[str, int, dict[str, float]]] = []
    seen: set[int] = set()

    for volume_id, metrics in reversed(ranked[-take_each_side:]):
        chosen.append(("best", volume_id, metrics))
        seen.add(volume_id)
    for volume_id, metrics in ranked[:take_each_side]:
        if volume_id not in seen:
            chosen.append(("worst", volume_id, metrics))
            seen.add(volume_id)
    if count % 2 == 1 and ranked:
        middle_volume_id, middle_metrics = ranked[len(ranked) // 2]
        if middle_volume_id not in seen:
            chosen.append(("median", middle_volume_id, middle_metrics))
    return chosen[:count]


def _representative_row(frame: pd.DataFrame, volume_id: int) -> pd.Series:
    subset = frame.loc[frame["volume"] == volume_id].copy()
    if subset.empty:
        raise ValueError(f"Could not find volume {volume_id} in evaluation CSV.")
    pixel_columns = [column for column in ("label0_pxl_cnt", "label1_pxl_cnt", "label2_pxl_cnt") if column in subset.columns]
    if pixel_columns:
        scores = subset[pixel_columns].sum(axis=1)
        return subset.loc[scores.idxmax()]
    if "slice" in subset.columns:
        midpoint = subset["slice"].median()
        return subset.iloc[(subset["slice"] - midpoint).abs().argmin()]
    return subset.iloc[len(subset) // 2]


def _export_qualitative_figures(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    data_config: dict[str, Any],
    split: str,
    device: torch.device,
    threshold: float,
    per_volume_metrics: dict[int, dict[str, float]],
    count: int,
    qualitative_dir: str | Path,
    region_tuned_configs: dict[str, dict[str, Any]] | None = None,
) -> Path | None:
    if count <= 0 or not per_volume_metrics:
        return None

    csv_path = data_config[f"{split}_csv"]
    dataset = ImagingCsvDataset(
        csv_path=csv_path,
        image_root=data_config.get("image_root"),
        image_col=data_config.get("image_col", "path"),
        label_cols=data_config.get("label_cols"),
        mask_col=data_config.get("mask_col"),
        image_size=tuple(data_config.get("image_size", (128, 128))),
        in_channels=int(data_config.get("in_channels", 1)),
        uncertain_value=float(data_config.get("uncertain_value", 0.0)),
        missing_label_value=float(data_config.get("missing_label_value", 0.0)),
        h5_image_key=data_config.get("h5_image_key", "image"),
        h5_mask_key=data_config.get("h5_mask_key", "mask"),
        mask_mode=data_config.get("mask_mode", "any"),
        mask_channel=data_config.get("mask_channel"),
        image_normalization=data_config.get("image_normalization", "auto"),
        slice_context=int(data_config.get("slice_context", 1)),
        slice_context_layout=data_config.get("slice_context_layout", "channels"),
    )
    frame = dataset.frame
    qualitative_dir = Path(qualitative_dir)
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    selected = _pick_qualitative_volumes(per_volume_metrics, count)
    summary: list[dict[str, Any]] = []
    colors = {
        "NCR": (255, 100, 80),
        "ED": (80, 220, 120),
        "ET": (80, 150, 255),
        "WT": (255, 180, 60),
        "TC": (220, 120, 255),
    }
    reference_bank = _build_counterfactual_reference_bank(model, dataset, device)
    sample_cache: dict[int, tuple[dict[str, Any], dict[str, torch.Tensor]]] = {}

    def _sample_outputs(row_index: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if row_index not in sample_cache:
            cached_sample = dataset[row_index]
            with torch.no_grad():
                cached_outputs = model(cached_sample["image"].unsqueeze(0).to(device))
            sample_cache[row_index] = (cached_sample, cached_outputs)
        return sample_cache[row_index]

    model.eval()
    for rank, volume_id, metrics in selected:
        row = _representative_row(frame, volume_id)
        row_index = int(row.name)
        sample, outputs = _sample_outputs(row_index)
        image = sample["image"]
        mask = sample["mask"]
        probs = torch.sigmoid(outputs["seg_logits"][0].cpu())
        pred_mask = (probs >= threshold).float()
        tuned_regions = _tuned_region_masks(
            probs,
            data_config.get("brats_region_channels") or {"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]},
            region_tuned_configs,
        )

        base_channel = _project_image_channels(image)
        factual_reconstruction = outputs["reconstruction"][0].cpu()
        factual_recon_channel = _project_image_channels(factual_reconstruction)
        gt_regions = _region_masks(mask)
        pred_regions = _region_masks(pred_mask)
        pred_regions.update(tuned_regions)
        context_shift_map = torch.zeros_like(base_channel)
        do_effect_map = torch.zeros_like(base_channel)
        context_cf_channel = factual_recon_channel
        disease_cf_channel = factual_recon_channel
        context_cf_regions: dict[str, torch.Tensor] | None = None
        disease_cf_regions: dict[str, torch.Tensor] | None = None
        context_reference: dict[str, Any] | None = None
        disease_reference: dict[str, Any] | None = None

        context_entry, disease_entry = _select_counterfactual_references(
            reference_bank,
            outputs["z_d"][0].detach().cpu(),
            outputs["z_c"][0].detach().cpu(),
            int(volume_id),
        )
        reference_contexts = [entry["z_c"] for entry in reference_bank if int(entry["volume"]) != int(volume_id)]
        if not reference_contexts:
            reference_contexts = [entry["z_c"] for entry in reference_bank]
        context_bank = torch.stack(reference_contexts, dim=0).to(device) if reference_contexts else None

        if context_entry is not None:
            context_reference = {
                "volume": int(context_entry["volume"]),
                "row_index": int(context_entry["row_index"]),
                "slice": int(context_entry["slice"]),
                "path": str(context_entry["path"]),
            }
            _, context_outputs = _sample_outputs(int(context_entry["row_index"]))
            with torch.no_grad():
                context_cf_outputs = model.refresh_outputs(outputs, context_outputs["z_c"])
            context_cf_probs = torch.sigmoid(context_cf_outputs["seg_logits"][0].cpu())
            context_cf_regions = _region_masks((context_cf_probs >= threshold).float())
            context_cf_channel = _project_image_channels(context_cf_outputs["reconstruction"][0].cpu())
            context_shift_map = _region_effect_map(probs, context_cf_probs)

        if disease_entry is not None:
            disease_reference = {
                "volume": int(disease_entry["volume"]),
                "row_index": int(disease_entry["row_index"]),
                "slice": int(disease_entry["slice"]),
                "path": str(disease_entry["path"]),
            }
            _, disease_outputs = _sample_outputs(int(disease_entry["row_index"]))
            with torch.no_grad():
                disease_cf_outputs = model.segment_outputs_from_parts(
                    disease_outputs["z_d"],
                    outputs["z_c"],
                    disease_outputs.get("disease_features"),
                )
                disease_cf_image = model.decode(disease_outputs["z_d"], outputs["z_c"])[0].cpu()
                disease_cf_probs = torch.sigmoid(disease_cf_outputs["seg_logits"][0].cpu())
                if context_bank is not None:
                    adjusted_factual = backdoor_adjusted_seg_logits(
                        model,
                        outputs.get("disease_features"),
                        outputs["z_d"],
                        outputs["z_c"],
                        max_contexts=min(8, int(context_bank.shape[0])),
                        context_bank=context_bank,
                    )
                    adjusted_disease = backdoor_adjusted_seg_logits(
                        model,
                        disease_outputs.get("disease_features"),
                        disease_outputs["z_d"],
                        outputs["z_c"],
                        max_contexts=min(8, int(context_bank.shape[0])),
                        context_bank=context_bank,
                    )
                    do_effect_map = _region_effect_map(
                        torch.sigmoid(adjusted_factual[0].cpu()),
                        torch.sigmoid(adjusted_disease[0].cpu()),
                    )
                else:
                    do_effect_map = _region_effect_map(probs, disease_cf_probs)
            disease_cf_regions = _region_masks((disease_cf_probs >= threshold).float())
            disease_cf_channel = _project_image_channels(disease_cf_image)

        panel_size = 176
        panels: list[Image.Image] = []
        for modality_index in range(4):
            if modality_index < image.shape[0]:
                panels.append(_channel_panel(image[modality_index], panel_size, f"Mod {modality_index + 1}"))
            else:
                panels.append(_blank_panel(panel_size, f"Mod {modality_index + 1}"))

        panels.extend(
            [
                _composite_overlay_panel(base_channel, gt_regions, panel_size, "GT Composite", colors),
                _composite_overlay_panel(base_channel, pred_regions, panel_size, "Pred Composite", colors),
                _channel_panel(factual_recon_channel, panel_size, "Factual Recon"),
                _channel_panel(context_cf_channel, panel_size, "Context CF Image"),
                (
                    _composite_overlay_panel(context_cf_channel, context_cf_regions, panel_size, "Context CF Seg", colors)
                    if context_cf_regions is not None
                    else _blank_panel(panel_size, "Context CF Seg")
                ),
                _channel_panel(disease_cf_channel, panel_size, "Disease CF Image"),
                (
                    _composite_overlay_panel(disease_cf_channel, disease_cf_regions, panel_size, "Disease CF Seg", colors)
                    if disease_cf_regions is not None
                    else _blank_panel(panel_size, "Disease CF Seg")
                ),
                _heatmap_panel(base_channel, do_effect_map, panel_size, "Do-Effect Map"),
                _heatmap_panel(base_channel, context_shift_map, panel_size, "Context Shift Map", color=(120, 210, 255)),
            ]
        )

        grid_columns = 4
        grid_rows = math.ceil(len(panels) / grid_columns)
        margin = 10
        header_height = 32
        canvas = Image.new(
            "RGB",
            (
                grid_columns * panel_size + (grid_columns + 1) * margin,
                grid_rows * panel_size + (grid_rows + 1) * margin + header_height,
            ),
            color=(18, 18, 18),
        )
        draw = ImageDraw.Draw(canvas)
        header = (
            f"{rank.upper()} volume {volume_id} slice {sample.get('slice', row.get('slice', 'n/a'))} "
            f"WT={metrics.get('brats/WT/dice', float('nan')):.3f} "
            f"TC={metrics.get('brats/TC/dice', float('nan')):.3f} "
            f"ET={metrics.get('brats/ET/dice', float('nan')):.3f} "
            f"thr={threshold:.2f}"
        )
        draw.text((margin, 8), header, fill=(240, 240, 240))

        for index, panel in enumerate(panels):
            row_index = index // grid_columns
            column_index = index % grid_columns
            x = margin + column_index * (panel_size + margin)
            y = header_height + margin + row_index * (panel_size + margin)
            canvas.paste(panel, (x, y))

        output_path = qualitative_dir / f"{rank}_volume_{volume_id}_slice_{sample.get('slice', row.get('slice', 'na'))}.png"
        canvas.save(output_path)
        summary.append(
            {
                "group": rank,
                "volume": int(volume_id),
                "slice": int(sample.get("slice", row.get("slice", -1))),
                "path": sample["path"],
                "figure": str(output_path),
                "metrics": metrics,
                "counterfactuals": {
                    "context_reference": context_reference,
                    "disease_reference": disease_reference,
                    "effect_mean": float(do_effect_map.mean().item()),
                    "context_shift_mean": float(context_shift_map.mean().item()),
                },
            }
        )

    save_json(summary, qualitative_dir / "summary.json")
    return qualitative_dir


def evaluate(
    checkpoint_path: str | Path,
    split: str = "val",
    config_path: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_batches: int | None = None,
    device_name: str = "auto",
    threshold: float = 0.5,
    threshold_sweep: list[float] | None = None,
    qualitative_count: int = 0,
    qualitative_dir: str | Path | None = None,
    tune_brats_regions: bool = False,
    region_threshold_sweep: list[float] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = _load_config(checkpoint, config_path)

    data_config = dict(config["data"])
    train_config = dict(config.get("training", {}))
    data_config["batch_size"] = batch_size or train_config.get("batch_size", 8)
    data_config["num_workers"] = num_workers if num_workers is not None else train_config.get("num_workers", 0)

    base_threshold = round(float(threshold), 4)
    sweep_thresholds = _parse_threshold_sweep(",".join(str(value) for value in threshold_sweep)) if threshold_sweep else []
    all_thresholds = sorted({base_threshold, *sweep_thresholds})
    region_tuning_thresholds: list[float] = []
    region_postprocess_candidates: dict[str, list[dict[str, Any]]] | None = None
    if tune_brats_regions and data_config.get("mask_mode") == "brats_subregions":
        explicit_region_thresholds = (
            _parse_threshold_sweep(",".join(str(value) for value in region_threshold_sweep))
            if region_threshold_sweep
            else []
        )
        region_tuning_thresholds = sorted({base_threshold, *(explicit_region_thresholds or sweep_thresholds or [base_threshold])})
        region_postprocess_candidates = _resolve_brats_region_postprocess_candidates(data_config)

    device = resolve_device(device_name if device_name != "auto" else train_config.get("device", "auto"))
    loader = make_dataloader(data_config, split, shuffle=False)
    iterator = islice(loader, max_batches) if max_batches is not None else loader
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cls_logits: list[torch.Tensor] = []
    cls_targets: list[torch.Tensor] = []
    loss_logs: list[dict[str, float]] = []
    loss_weights: list[float] = []
    seg_logs: list[dict[str, float]] = []
    seg_weights: list[float] = []
    volume_tracker = (
        VolumeMetricTracker(
            all_thresholds,
            data_config,
            region_thresholds=region_tuning_thresholds,
            region_postprocess_candidates=region_postprocess_candidates,
        )
        if "mask" in loader.dataset[0] and "volume" in loader.dataset[0]
        else None
    )

    with torch.no_grad():
        for batch in tqdm(iterator, total=total_batches, leave=False):
            batch_size_actual = int(batch["image"].shape[0])
            batch_device = move_batch_to_device(batch, device)
            outputs = model(batch_device["image"])
            _, logs = compute_crn_losses(model, batch_device, outputs, config.get("loss", {}))
            loss_logs.append({key: float(value.detach().cpu()) for key, value in logs.items()})
            loss_weights.append(batch_size_actual)

            if "logits" in outputs and "label" in batch_device:
                cls_logits.append(outputs["logits"].detach().cpu())
                cls_targets.append(batch_device["label"].detach().cpu())

            if "seg_logits" in outputs and "mask" in batch_device:
                seg_logits_cpu = outputs["seg_logits"].detach().cpu()
                seg_targets_cpu = batch_device["mask"].detach().cpu()
                seg_logs.append(_segmentation_metrics(seg_logits_cpu, seg_targets_cpu, data_config, base_threshold))
                seg_weights.append(batch_size_actual)
                if volume_tracker is not None:
                    volume_tracker.add_batch(seg_logits_cpu, seg_targets_cpu, batch.get("volume"), batch.get("slice"))

    metrics: dict[str, Any] = _weighted_average_dicts(loss_logs, loss_weights)
    if cls_logits:
        metrics.update(classification_metrics(torch.cat(cls_logits, dim=0), torch.cat(cls_targets, dim=0), base_threshold))
    if seg_logs:
        metrics.update(_weighted_average_dicts(seg_logs, seg_weights))

    per_volume_threshold_metrics: dict[float, dict[int, dict[str, float]]] = {}
    region_tuning: dict[str, Any] | None = None
    if volume_tracker is not None:
        volume_summaries, per_volume_threshold_metrics, region_tuning = volume_tracker.finalize()
        base_volume_metrics = volume_summaries.get(base_threshold, {})
        metrics.update(_prefix_metrics(base_volume_metrics, "volume"))

        if sweep_thresholds:
            sweep_summary = {
                f"{threshold_value:.4f}": {
                    "score": _threshold_score(summary),
                    "metrics": _prefix_metrics(summary, "volume"),
                }
                for threshold_value, summary in volume_summaries.items()
            }
            best_threshold = max(all_thresholds, key=lambda candidate: _threshold_score(volume_summaries.get(candidate, {})))
            best_score = _threshold_score(volume_summaries.get(best_threshold, {}))
            metrics["sweep/best_threshold"] = best_threshold
            metrics["sweep/best_volume_score"] = best_score
            metrics.update(_prefix_metrics(volume_summaries[best_threshold], "sweep_best_volume"))

            checkpoint_path_obj = Path(checkpoint_path)
            sweep_output = checkpoint_path_obj.with_name(f"{checkpoint_path_obj.stem}_{split}_threshold_sweep.json")
            save_json(
                {
                    "thresholds": all_thresholds,
                    "best_threshold": best_threshold,
                    "best_volume_score": best_score,
                    "results": sweep_summary,
                },
                sweep_output,
            )
            qualitative_threshold = best_threshold
        else:
            qualitative_threshold = base_threshold
    else:
        qualitative_threshold = base_threshold

    if region_tuning is not None:
        metrics.update(_prefix_metrics(region_tuning["summary"], "region_tuned"))
        for region_name, config in region_tuning["selected_configs"].items():
            metrics[f"region_tuned/config/{region_name}/threshold"] = float(config["threshold"])
            metrics[f"region_tuned/config/{region_name}/min_component_size"] = int(config.get("min_component_size", 0))
            metrics[f"region_tuned/config/{region_name}/fill_holes"] = bool(config.get("fill_holes", False))
            metrics[f"region_tuned/config/{region_name}/keep_largest"] = bool(config.get("keep_largest", False))
        checkpoint_path_obj = Path(checkpoint_path)
        region_tuning_output = checkpoint_path_obj.with_name(f"{checkpoint_path_obj.stem}_{split}_region_tuning.json")
        save_json(
            {
                "thresholds": region_tuning_thresholds,
                "selected_configs": region_tuning["selected_configs"],
                "summary": region_tuning["summary"],
                "candidates": region_tuning["candidates"],
            },
            region_tuning_output,
        )
        metrics["region_tuned/output_path"] = str(region_tuning_output)

    metrics["eval/epoch"] = int(checkpoint.get("epoch", -1))
    metrics["eval/split"] = split
    metrics["eval/checkpoint"] = str(Path(checkpoint_path))
    metrics["eval/threshold_used"] = base_threshold

    if volume_tracker is not None and qualitative_count > 0:
        if qualitative_dir is None:
            checkpoint_path_obj = Path(checkpoint_path)
            qualitative_dir = checkpoint_path_obj.with_name(f"{checkpoint_path_obj.stem}_{split}_qualitative")
        qualitative_metrics = (
            region_tuning["per_volume"]
            if region_tuning is not None and region_tuning.get("per_volume")
            else per_volume_threshold_metrics.get(qualitative_threshold, {})
        )
        exported_dir = _export_qualitative_figures(
            checkpoint=checkpoint,
            model=model,
            data_config=data_config,
            split=split,
            device=device,
            threshold=qualitative_threshold,
            per_volume_metrics=qualitative_metrics,
            count=qualitative_count,
            qualitative_dir=qualitative_dir,
            region_tuned_configs=region_tuning["selected_configs"] if region_tuning is not None else None,
        )
        if exported_dir is not None:
            metrics["qualitative/threshold_used"] = qualitative_threshold
            metrics["qualitative/output_dir"] = str(exported_dir)
            if region_tuning is not None:
                metrics["qualitative/uses_region_tuning"] = True

    if output_path is None:
        checkpoint_path_obj = Path(checkpoint_path)
        output_path = checkpoint_path_obj.with_name(f"{checkpoint_path_obj.stem}_{split}_metrics.json")
    save_json(metrics, output_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CRN checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint, e.g. runs/brats_crn/best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to evaluate.")
    parser.add_argument("--config", help="Optional config path. Uses checkpoint config by default.")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, help="Optional num_workers override.")
    parser.add_argument("--max-batches", type=int, help="Evaluate only this many batches.")
    parser.add_argument("--device", default="auto", help="Device override: auto, cpu, cuda, mps.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Base probability threshold for metrics.")
    parser.add_argument(
        "--threshold-sweep",
        help="Optional comma-separated thresholds, e.g. 0.3,0.35,0.4,0.45,0.5",
    )
    parser.add_argument("--qualitative-count", type=int, default=0, help="Number of qualitative volume figures to export.")
    parser.add_argument("--qualitative-dir", help="Directory for qualitative figures. Defaults next to the checkpoint.")
    parser.add_argument(
        "--tune-brats-regions",
        action="store_true",
        help="Independently tune WT/TC/ET thresholds with simple 3D post-processing on volume metrics.",
    )
    parser.add_argument(
        "--region-threshold-sweep",
        help="Optional comma-separated thresholds for WT/TC/ET tuning. Defaults to --threshold-sweep.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        split=args.split,
        config_path=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        device_name=args.device,
        threshold=args.threshold,
        threshold_sweep=_parse_threshold_sweep(args.threshold_sweep),
        qualitative_count=args.qualitative_count,
        qualitative_dir=args.qualitative_dir,
        tune_brats_regions=args.tune_brats_regions,
        region_threshold_sweep=_parse_threshold_sweep(args.region_threshold_sweep),
        output_path=args.output,
    )
    print(metrics)


if __name__ == "__main__":
    main()
