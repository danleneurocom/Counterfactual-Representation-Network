from __future__ import annotations

import argparse
from itertools import islice
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from tqdm import tqdm

from crn.data import make_dataloader
from crn.losses import CounterfactualMemory, compute_crn_losses
from crn.metrics import (
    binary_segmentation_metrics,
    brats_region_metrics,
    brats_volume_metrics_from_probs,
    multilabel_segmentation_metrics,
    multilabel_volume_metrics_from_probs,
)
from crn.models import CounterfactualRepresentationNetwork
from crn.utils import (
    load_yaml,
    move_batch_to_device,
    resolve_device,
    save_json,
    seed_everything,
)


def _mean_logs(logs: list[dict[str, Tensor]]) -> dict[str, float]:
    if not logs:
        return {}
    keys = logs[0].keys()
    return {key: float(torch.stack([entry[key].cpu() for entry in logs]).mean()) for key in keys}


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
    values = [chunk.strip() for chunk in str(spec).split(",") if chunk.strip()]
    thresholds = sorted({round(float(value), 4) for value in values})
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold sweep values must be between 0 and 1, got {threshold}")
    return thresholds


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _segmentation_metrics(logits: Tensor, target: Tensor, data_config: dict[str, Any], threshold: float) -> dict[str, float]:
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


def _volume_segmentation_metrics(probs: Tensor, target: Tensor, data_config: dict[str, Any], threshold: float) -> dict[str, float]:
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


def _dataset_volume_ids(dataset: Any) -> list[int]:
    frame = getattr(dataset, "frame", None)
    if frame is None or "volume" not in frame.columns:
        return []
    return [int(volume_id) for volume_id in frame["volume"].drop_duplicates().tolist()]


def _build_counterfactual_memory(
    dataset: Any,
    model: CounterfactualRepresentationNetwork,
    train_config: dict[str, Any],
    device: torch.device,
) -> CounterfactualMemory | None:
    memory_config = dict(train_config.get("counterfactual_memory") or {})
    if not bool(memory_config.get("enabled", False)):
        return None
    volume_ids = _dataset_volume_ids(dataset)
    if not volume_ids:
        return None
    return CounterfactualMemory(
        num_samples=len(dataset),
        volume_ids=volume_ids,
        latent_dim=model.latent_dim,
        device=device,
        volume_momentum=float(memory_config.get("volume_momentum", 0.95)),
        sample_momentum=float(memory_config.get("sample_momentum", 0.90)),
        match_topk=int(memory_config.get("match_topk", 32)),
        min_context_distance=float(memory_config.get("min_context_distance", 0.10)),
        exemplar_capacity=int(memory_config.get("exemplar_capacity", 32)),
        mixture_topk=int(memory_config.get("mixture_topk", 4)),
        mixture_temperature=float(memory_config.get("mixture_temperature", 0.35)),
    )


def _apply_volume_context(
    model: CounterfactualRepresentationNetwork,
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    counterfactual_memory: CounterfactualMemory | None,
) -> dict[str, Tensor]:
    if counterfactual_memory is None or "volume" not in batch or "z_c" not in outputs:
        return outputs
    z_c_slice = outputs.get("z_c_slice", outputs["z_c"])
    volume_context = counterfactual_memory.lookup_volume_context(batch.get("volume"), z_c_slice)
    fused_context = model.fuse_context_latents(z_c_slice, volume_context)
    return model.refresh_outputs(outputs, fused_context, volume_context=volume_context)


class VolumeMetricTracker:
    def __init__(self, thresholds: list[float], data_config: dict[str, Any]) -> None:
        self.thresholds = thresholds
        self.data_config = data_config
        self.summary_by_threshold: dict[float, list[dict[str, float]]] = {threshold: [] for threshold in thresholds}
        self.current_volume: int | None = None
        self.current_slices: list[int] = []
        self.current_probs: list[Tensor] = []
        self.current_targets: list[Tensor] = []
        self.seen_volumes: set[int] = set()

    def add_batch(
        self,
        logits: Tensor,
        target: Tensor,
        volumes: Tensor | None,
        slices: Tensor | None,
    ) -> None:
        if volumes is None or slices is None:
            return
        probs = torch.sigmoid(logits.detach()).cpu().to(torch.float16)
        targets = target.detach().cpu().to(torch.uint8)
        volume_ids = volumes.detach().cpu().tolist()
        slice_ids = slices.detach().cpu().tolist()
        for index, volume_id in enumerate(volume_ids):
            self._add_sample(probs[index], targets[index], int(volume_id), int(slice_ids[index]))

    def finalize(self) -> dict[float, dict[str, float]]:
        self._flush_current()
        return {
            threshold: _weighted_average_dicts(items, [1.0] * len(items)) if items else {}
            for threshold, items in self.summary_by_threshold.items()
        }

    def _add_sample(self, probs: Tensor, target: Tensor, volume_id: int, slice_id: int) -> None:
        if self.current_volume is None:
            self._start_volume(volume_id)
        elif volume_id != self.current_volume:
            self._flush_current()
            if volume_id in self.seen_volumes:
                raise ValueError(
                    "Volume IDs are not grouped in validation order. Ensure the validation CSV is sorted by volume and slice."
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
        self.seen_volumes.add(self.current_volume)
        self.current_volume = None
        self.current_slices = []
        self.current_probs = []
        self.current_targets = []


def _loss_warmup_factor(epoch: int, warmup_config: dict[str, Any] | None) -> float:
    if not warmup_config:
        return 1.0
    warmup_epochs = int(warmup_config.get("epochs", 0))
    if warmup_epochs <= 0:
        return 1.0
    start_factor = float(warmup_config.get("start_factor", 0.0))
    if warmup_epochs == 1:
        return 1.0
    progress = min(1.0, max(0.0, float(epoch - 1) / float(warmup_epochs - 1)))
    return start_factor + (1.0 - start_factor) * progress


def _effective_loss_config(
    loss_config: dict[str, Any],
    epoch: int,
    warmup_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], float]:
    if not warmup_config:
        return dict(loss_config), 1.0
    factor = _loss_warmup_factor(epoch, warmup_config)
    warmup_keys = warmup_config.get("keys") or [
        "lambda_dis",
        "lambda_adjustment",
        "lambda_cf_stability",
        "lambda_region_adjustment",
        "lambda_seg_adjustment",
        "lambda_region_cf_stability",
        "lambda_region_cf_contrastive",
        "lambda_seg_cf_stability",
        "lambda_region_disease_swap",
        "lambda_seg_disease_swap",
        "lambda_disease_swap",
    ]
    effective = dict(loss_config)
    for key in warmup_keys:
        if key in effective:
            effective[key] = float(effective[key]) * factor
    return effective, factor


def _comparison_value(value: float | None, mode: str) -> float:
    if value is None or not math.isfinite(float(value)):
        return float("-inf") if mode == "max" else float("inf")
    return float(value)


def _is_better_metric(candidate: float | None, best: float | None, mode: str, eps: float = 1e-8) -> bool:
    candidate_value = _comparison_value(candidate, mode)
    best_value = _comparison_value(best, mode)
    if mode == "max":
        return candidate_value > best_value + eps
    if mode == "min":
        return candidate_value < best_value - eps
    raise ValueError(f"Unknown checkpoint mode: {mode}")


def _should_update_best(
    metrics: dict[str, float],
    best_primary: float | None,
    best_secondary: float | None,
    train_config: dict[str, Any],
) -> tuple[bool, float | None, float | None]:
    primary_key = str(train_config.get("checkpoint_metric", "loss/total"))
    primary_mode = str(train_config.get("checkpoint_mode", "min"))
    secondary_key = train_config.get("checkpoint_tiebreak_metric")
    secondary_mode = str(train_config.get("checkpoint_tiebreak_mode", "min"))
    primary_value = metrics.get(primary_key)
    secondary_value = metrics.get(str(secondary_key)) if secondary_key else None

    if best_primary is None:
        return True, primary_value, secondary_value
    if _is_better_metric(primary_value, best_primary, primary_mode):
        return True, primary_value, secondary_value
    if _is_better_metric(best_primary, primary_value, primary_mode):
        return False, primary_value, secondary_value
    if secondary_key and _is_better_metric(secondary_value, best_secondary, secondary_mode):
        return True, primary_value, secondary_value
    return False, primary_value, secondary_value


def _inflate_input_kernel(source: Tensor, target_shape: torch.Size) -> Tensor | None:
    if source.ndim == 4 and len(target_shape) == 5:
        if source.shape[0] != target_shape[0] or source.shape[1] != target_shape[1] or source.shape[-2:] != target_shape[-2:]:
            return None
        depth = int(target_shape[2])
        return source.unsqueeze(2).repeat(1, 1, depth, 1, 1) / float(depth)
    if source.ndim != 4 or len(target_shape) != 4:
        return None
    if source.shape[0] != target_shape[0] or source.shape[2:] != target_shape[2:]:
        return None
    if target_shape[1] % source.shape[1] != 0:
        return None
    repeats = target_shape[1] // source.shape[1]
    return source.repeat(1, repeats, 1, 1) / float(repeats)


def _load_compatible_state_dict(model: torch.nn.Module, state_dict: dict[str, Tensor]) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    compatible: dict[str, Tensor] = {}
    adapted_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if model_state[key].shape == value.shape:
            compatible[key] = value
            continue
        inflated = _inflate_input_kernel(value, model_state[key].shape)
        if inflated is not None:
            compatible[key] = inflated
            adapted_keys.append(key)
            continue
        skipped_keys.append(key)

    incompatible = model.load_state_dict(compatible, strict=False)
    missing_keys = list(incompatible.missing_keys)
    return missing_keys, list(incompatible.unexpected_keys), adapted_keys + skipped_keys


def run_epoch(
    model: CounterfactualRepresentationNetwork,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW | None,
    loss_config: dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
    counterfactual_memory: CounterfactualMemory | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    epoch_logs: list[dict[str, Tensor]] = []
    iterator = islice(loader, max_batches) if max_batches is not None else loader
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(iterator, total=total_batches, leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"])
            if counterfactual_memory is not None:
                counterfactual_memory.update(batch.get("index"), batch.get("volume"), outputs["z_d"], outputs["z_c_slice"])
                outputs = _apply_volume_context(model, outputs, batch, counterfactual_memory)
                counterfactual_memory.store_exemplars(
                    batch.get("index"),
                    batch.get("volume"),
                    outputs["z_d"],
                    outputs["z_c"],
                    mask=batch.get("mask"),
                    label=batch.get("label"),
                    disease_features=outputs.get("disease_features"),
                )
            loss, logs = compute_crn_losses(model, batch, outputs, loss_config, counterfactual_memory=counterfactual_memory)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            epoch_logs.append(logs)
    return _mean_logs(epoch_logs)


def validate_epoch(
    model: CounterfactualRepresentationNetwork,
    loader: torch.utils.data.DataLoader,
    loss_config: dict[str, Any],
    data_config: dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
    base_threshold: float = 0.5,
    threshold_sweep: list[float] | None = None,
    counterfactual_memory: CounterfactualMemory | None = None,
) -> dict[str, float]:
    model.eval()
    epoch_logs: list[dict[str, Tensor]] = []
    seg_logs: list[dict[str, float]] = []
    seg_weights: list[float] = []
    all_thresholds = sorted({round(float(base_threshold), 4), *(threshold_sweep or [])})
    iterator = islice(loader, max_batches) if max_batches is not None else loader
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    volume_tracker = (
        VolumeMetricTracker(all_thresholds, data_config)
        if "mask" in loader.dataset[0] and "volume" in loader.dataset[0]
        else None
    )
    if volume_tracker is not None and max_batches is not None:
        raise ValueError(
            "Batch-limited validation is unsafe for volume metrics because it can cut a volume mid-case. "
            "Use full validation for BraTS checkpointing."
        )

    with torch.no_grad():
        for batch in tqdm(iterator, total=total_batches, leave=False):
            batch_size_actual = int(batch["image"].shape[0])
            batch_device = move_batch_to_device(batch, device)
            outputs = model(batch_device["image"])
            if counterfactual_memory is not None:
                counterfactual_memory.update(
                    batch_device.get("index"),
                    batch_device.get("volume"),
                    outputs["z_d"],
                    outputs["z_c_slice"],
                )
                outputs = _apply_volume_context(model, outputs, batch_device, counterfactual_memory)
                counterfactual_memory.store_exemplars(
                    batch_device.get("index"),
                    batch_device.get("volume"),
                    outputs["z_d"],
                    outputs["z_c"],
                    mask=batch_device.get("mask"),
                    label=batch_device.get("label"),
                    disease_features=outputs.get("disease_features"),
                )
            _, logs = compute_crn_losses(
                model,
                batch_device,
                outputs,
                loss_config,
                counterfactual_memory=counterfactual_memory,
            )
            epoch_logs.append(logs)

            if "seg_logits" in outputs and "mask" in batch_device:
                seg_logits_cpu = outputs["seg_logits"].detach().cpu()
                seg_targets_cpu = batch_device["mask"].detach().cpu()
                seg_logs.append(_segmentation_metrics(seg_logits_cpu, seg_targets_cpu, data_config, base_threshold))
                seg_weights.append(batch_size_actual)
                if volume_tracker is not None:
                    volume_tracker.add_batch(seg_logits_cpu, seg_targets_cpu, batch.get("volume"), batch.get("slice"))

    metrics = _mean_logs(epoch_logs)
    if seg_logs:
        metrics.update(_weighted_average_dicts(seg_logs, seg_weights))
    if volume_tracker is not None:
        volume_summaries = volume_tracker.finalize()
        metrics.update(_prefix_metrics(volume_summaries.get(base_threshold, {}), "volume"))
        best_threshold = max(all_thresholds, key=lambda candidate: _threshold_score(volume_summaries.get(candidate, {})))
        best_score = _threshold_score(volume_summaries.get(best_threshold, {}))
        metrics["sweep/best_threshold"] = float(best_threshold)
        metrics["sweep/best_volume_score"] = float(best_score)
        metrics.update(_prefix_metrics(volume_summaries[best_threshold], "sweep_best_volume"))
    return metrics


def build_model(config: dict[str, Any]) -> CounterfactualRepresentationNetwork:
    data_config = config["data"]
    model_config = dict(config["model"])
    slice_context = int(data_config.get("slice_context", 1))
    slice_context_layout = str(data_config.get("slice_context_layout", "channels")).lower()
    if slice_context_layout == "depth":
        model_config.setdefault("backbone_mode", "2.5d")
        model_config.setdefault("in_channels", int(data_config.get("in_channels", 1)))
    else:
        model_config.setdefault("in_channels", int(data_config.get("in_channels", 1)) * slice_context)
    model_config.setdefault("image_size", data_config.get("image_size", (128, 128)))
    model_config.setdefault("num_classes", len(data_config.get("label_cols") or []))
    model_config.setdefault("num_seg_classes", 1 if data_config.get("mask_col") else 0)
    return CounterfactualRepresentationNetwork(**model_config)


def train(config: dict[str, Any]) -> None:
    seed_everything(int(config.get("seed", 7)))
    train_config = config["training"]
    data_config = dict(config["data"])
    data_config.update(
        {
            "batch_size": train_config.get("batch_size", 8),
            "num_workers": train_config.get("num_workers", 0),
        }
    )
    device = resolve_device(train_config.get("device", "auto"))
    output_dir = Path(train_config.get("output_dir", "runs/crn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.resolved.json")

    train_loader = make_dataloader(data_config, "train", shuffle=True)
    val_loader = make_dataloader(data_config, "val", shuffle=False)
    model = build_model(config).to(device)
    init_checkpoint = train_config.get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        if bool(train_config.get("init_strict", False)):
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=True)
            ignored = []
        else:
            missing, unexpected, ignored = _load_compatible_state_dict(model, checkpoint["model"])
        print(
            {
                "warmstart": str(init_checkpoint),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "ignored_or_adapted_keys": len(ignored),
            }
        )
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 3e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    train_counterfactual_memory = _build_counterfactual_memory(train_loader.dataset, model, train_config, device)
    val_counterfactual_memory_template = _build_counterfactual_memory(val_loader.dataset, model, train_config, device)

    best_primary: float | None = None
    best_secondary: float | None = None
    epochs = int(train_config.get("epochs", 1))
    base_threshold = float(train_config.get("checkpoint_threshold", 0.5))
    threshold_sweep = _parse_threshold_sweep(train_config.get("checkpoint_threshold_sweep"))
    loss_warmup = train_config.get("loss_warmup")
    for epoch in range(1, epochs + 1):
        loss_config_epoch, warmup_factor = _effective_loss_config(config.get("loss", {}), epoch, loss_warmup)
        train_logs = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_config_epoch,
            device,
            train_config.get("max_train_batches"),
            counterfactual_memory=train_counterfactual_memory,
        )
        val_counterfactual_memory = None
        if val_counterfactual_memory_template is not None:
            val_counterfactual_memory_template.reset()
            val_counterfactual_memory = val_counterfactual_memory_template
        val_logs = validate_epoch(
            model,
            val_loader,
            loss_config_epoch,
            data_config,
            device,
            train_config.get("max_val_batches"),
            base_threshold=base_threshold,
            threshold_sweep=threshold_sweep,
            counterfactual_memory=val_counterfactual_memory,
        )
        val_loss = val_logs.get("loss/total", float("inf"))
        best_update, primary_value, secondary_value = _should_update_best(val_logs, best_primary, best_secondary, train_config)

        log_line = {
            "epoch": epoch,
            "train_loss": train_logs.get("loss/total"),
            "val_loss": val_loss,
            "warmup_factor": warmup_factor,
        }
        checkpoint_metric_key = str(train_config.get("checkpoint_metric", "loss/total"))
        if checkpoint_metric_key in val_logs:
            log_line["checkpoint_metric"] = {checkpoint_metric_key: val_logs[checkpoint_metric_key]}
        print(log_line)
        save_json(
            {
                "epoch": epoch,
                "warmup_factor": warmup_factor,
                "train": train_logs,
                "val": val_logs,
            },
            output_dir / f"epoch_{epoch:03d}.json",
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val_loss": val_loss,
            "monitor_metric": checkpoint_metric_key,
            "monitor_value": primary_value,
            "monitor_tiebreak_metric": train_config.get("checkpoint_tiebreak_metric"),
            "monitor_tiebreak_value": secondary_value,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if best_update:
            best_primary = primary_value
            best_secondary = secondary_value
            torch.save(checkpoint, output_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Counterfactual Representation Network.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_yaml(args.config))


if __name__ == "__main__":
    main()
