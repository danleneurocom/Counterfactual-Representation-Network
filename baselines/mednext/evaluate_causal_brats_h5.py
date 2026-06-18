from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mednext.calibration import (
    CALIBRATION_OBJECTIVES,
    BratsRegionThresholdSweep,
    brats_region_metrics_from_plausibility_thresholds,
    brats_region_metrics_from_thresholds,
    parse_region_thresholds,
    parse_threshold_candidates,
    prefix_metrics,
)
from baselines.mednext.causal import build_causal_mednext
from baselines.mednext.common import mirror_tta_logits, parse_mirror_tta_axes
from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.evaluate_causal_brats_h5 import BraTSH5VolumeDataset, _context_overlap, _save_json, _segmentation_loss, _volume_metrics
from baselines.mednext.train_causal_utsw import _et_volume_veto_metric_item
from baselines.segformer3d.train_causal_utsw import build_context_bank, _prefix_metrics
from baselines.segformer3d.train_utsw import _average_metric_dicts, _mean, _resolve_device
from crn.metrics import brats_region_metrics
from crn.metrics import brats_structural_region_metrics, brats_structural_region_metrics_from_thresholds


def _make_loader(dataset: BraTSH5VolumeDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _logit(probability: Tensor) -> Tensor:
    probability = probability.clamp(1e-4, 1.0 - 1e-4)
    return torch.logit(probability)


def _case_metric_value(metrics: dict[str, float], name: str) -> float | None:
    value = metrics.get(name)
    return None if value is None else float(value)


def _batch_item_value(batch: dict[str, Any], key: str, index: int) -> Any:
    value = batch.get(key)
    if isinstance(value, Tensor):
        item = value[index]
        return int(item.detach().cpu()) if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


_CASE_RECORD_METRIC_KEYS = (
    "brats/mean_dice",
    "brats/ET/dice",
    "brats/TC/dice",
    "brats/WT/dice",
    "brats/pred_foreground_ratio",
    "brats/target_foreground_ratio",
    "brats/WT/pred_foreground_ratio",
    "brats/TC/pred_foreground_ratio",
    "brats/ET/pred_foreground_ratio",
    "brats/WT/target_foreground_ratio",
    "brats/TC/target_foreground_ratio",
    "brats/ET/target_foreground_ratio",
    "plausibility/low_threshold_fraction",
    "plausibility/low_threshold_count",
    "plausibility/low_stability_gate_fraction",
    "plausibility/tc_collapse_gate_fraction",
    "plausibility/base_WT_pred_foreground_ratio_mean",
    "plausibility/base_WT_pred_foreground_ratio_min",
    "plausibility/base_WT_pred_foreground_ratio_max",
    "plausibility/base_TC_pred_foreground_ratio_mean",
    "plausibility/stability_score_mean",
)


def _brats_case_record(
    batch: dict[str, Any],
    batch_index: int,
    target: Tensor,
    threshold: float,
    variants: dict[str, Tensor],
    *,
    region_thresholds: dict[str, float] | None = None,
    plausibility_region_thresholds: tuple[
        dict[str, float],
        dict[str, float],
        float,
        float,
        float,
        float,
        float,
    ]
    | None = None,
) -> dict[str, Any]:
    case_id = _batch_item_value(batch, "case_id", batch_index)
    record: dict[str, Any] = {
        "case_id": str(case_id),
        "volume": _batch_item_value(batch, "volume", batch_index),
        "path": str(_batch_item_value(batch, "path", batch_index)),
    }
    for variant_name, logits in variants.items():
        item_logits = logits[batch_index : batch_index + 1]
        item_target = target[batch_index : batch_index + 1]
        item_metrics = brats_region_metrics(item_logits.detach().cpu(), item_target.detach().cpu(), threshold=threshold)
        for key in _CASE_RECORD_METRIC_KEYS:
            value = _case_metric_value(item_metrics, key)
            if value is not None:
                record[f"{variant_name}/{key}"] = value
        if region_thresholds is not None:
            calibrated_metrics = brats_region_metrics_from_thresholds(item_logits, item_target, region_thresholds)
            for key in _CASE_RECORD_METRIC_KEYS:
                value = _case_metric_value(calibrated_metrics, key)
                if value is not None:
                    record[f"{variant_name}_region_calibrated/{key}"] = value
        if plausibility_region_thresholds is not None:
            (
                base_thresholds,
                low_thresholds,
                low_stability_wt_ratio_threshold,
                low_stability_threshold,
                tc_collapse_wt_ratio_min,
                tc_collapse_wt_ratio_max,
                tc_collapse_tc_ratio_threshold,
            ) = plausibility_region_thresholds
            plausibility_metrics = brats_region_metrics_from_plausibility_thresholds(
                item_logits,
                item_target,
                base_thresholds,
                low_thresholds,
                low_stability_wt_ratio_threshold=low_stability_wt_ratio_threshold,
                low_stability_threshold=low_stability_threshold,
                tc_collapse_wt_ratio_min=tc_collapse_wt_ratio_min,
                tc_collapse_wt_ratio_max=tc_collapse_wt_ratio_max,
                tc_collapse_tc_ratio_threshold=tc_collapse_tc_ratio_threshold,
            )
            for key in _CASE_RECORD_METRIC_KEYS:
                value = _case_metric_value(plausibility_metrics, key)
                if value is not None:
                    record[f"{variant_name}_plausibility_region_calibrated/{key}"] = value
    return record


def _should_build_context_bank(args: argparse.Namespace) -> bool:
    return (int(args.adjustment_contexts) > 0 or int(args.cct_contexts) > 0) and int(args.context_bank_size) > 0


def _cct_selected_contexts(
    context_bank: Tensor,
    max_contexts: int | None,
    *,
    anchor_context: Tensor | None = None,
    selection: str = "uniform",
) -> Tensor:
    if context_bank.ndim != 2:
        raise ValueError(f"context_bank must have shape [K, latent_dim], got {tuple(context_bank.shape)}")
    if max_contexts is None or int(max_contexts) <= 0 or context_bank.shape[0] <= int(max_contexts):
        return context_bank
    selection = str(selection).lower()
    context_count = int(max_contexts)
    if selection == "uniform" or anchor_context is None:
        positions = torch.linspace(0, context_bank.shape[0] - 1, steps=context_count, device=context_bank.device)
        return context_bank[positions.round().long()]

    if anchor_context.ndim != 2:
        raise ValueError(f"anchor_context must have shape [B, latent_dim], got {tuple(anchor_context.shape)}")
    if anchor_context.shape[1] != context_bank.shape[1]:
        raise ValueError(
            f"anchor_context latent dim {anchor_context.shape[1]} does not match context_bank dim {context_bank.shape[1]}"
        )

    anchor = anchor_context.detach().to(device=context_bank.device, dtype=context_bank.dtype)
    distances = torch.cdist(anchor.float(), context_bank.float()).mean(dim=0)
    if selection == "nearest":
        return context_bank[torch.topk(distances, k=context_count, largest=False, sorted=True).indices]
    if selection == "farthest":
        return context_bank[torch.topk(distances, k=context_count, largest=True, sorted=True).indices]
    if selection == "diverse-nearest":
        pool_count = min(int(context_bank.shape[0]), max(context_count, context_count * 4))
        pool = torch.topk(distances, k=pool_count, largest=False, sorted=True).indices
        positions = torch.linspace(0, pool_count - 1, steps=context_count, device=context_bank.device)
        return context_bank[pool[positions.round().long()]]
    raise ValueError(f"Unknown CCT context selection mode: {selection!r}")


def _cct_transport_outputs(
    model: torch.nn.Module,
    features: tuple[Tensor, ...],
    z_d: Tensor,
    z_c: Tensor,
    context_bank: Tensor,
    image: Tensor,
    *,
    max_contexts: int,
    selection: str,
    instability_scale: float,
) -> dict[str, Tensor]:
    bank = _cct_selected_contexts(
        context_bank.to(device=z_d.device, dtype=z_d.dtype),
        max_contexts,
        anchor_context=z_c,
        selection=selection,
    )
    if bank.numel() == 0:
        raise ValueError("CCT requested an empty context bank.")
    transported_probs: list[Tensor] = []
    for context in bank:
        z_c = context.unsqueeze(0).expand(z_d.shape[0], -1)
        transported_logits = model.segment_from_latents(features, z_d, z_c, image=image)
        transported_probs.append(torch.sigmoid(transported_logits))
    prob_stack = torch.stack(transported_probs, dim=0)
    consensus_prob = prob_stack.mean(dim=0)
    instability = prob_stack.var(dim=0, unbiased=False).clamp_min(0.0).sqrt()
    gated_prob = (consensus_prob - float(instability_scale) * instability).clamp(1e-4, 1.0 - 1e-4)
    return {
        "consensus_logits": _logit(consensus_prob),
        "stability_gated_logits": _logit(gated_prob),
        "instability": instability,
        "transported_context_count": torch.tensor(float(bank.shape[0]), device=z_d.device),
    }


def _build_model(checkpoint: dict[str, Any], args: argparse.Namespace):
    config = dict(checkpoint.get("config", {}))
    proxy_dims = dict(checkpoint.get("proxy_dims") or {})
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
        use_causal_mediator_router=bool(config.get("use_causal_mediator_router", False)),
        use_nested_causal_intervention=bool(config.get("use_nested_causal_intervention", False)),
        nested_causal_gate_scale=float(config.get("nested_causal_gate_scale", 1.0)),
        region_causal_bottleneck_scale=float(config.get("region_causal_bottleneck_scale", 0.0)),
        region_causal_background_leak=float(config.get("region_causal_background_leak", 0.05)),
        region_causal_base=str(config.get("region_causal_base", "prior")),
        region_causal_mask_source=str(config.get("region_causal_mask_source", "spatial")),
        region_volume_scale=float(
            args.region_volume_scale if args.region_volume_scale is not None else config.get("region_volume_scale", 1000.0)
        ),
        et_volume_veto_scale=float(
            args.et_volume_veto_scale
            if args.et_volume_veto_scale is not None
            else config.get("et_volume_veto_scale", 0.0)
        ),
        et_volume_veto_multiplier=float(
            args.et_volume_veto_multiplier
            if args.et_volume_veto_multiplier is not None
            else config.get("et_volume_veto_multiplier", 4.0)
        ),
        et_volume_veto_min_fraction=float(
            args.et_volume_veto_min_fraction
            if args.et_volume_veto_min_fraction is not None
            else config.get("et_volume_veto_min_fraction", 5e-4)
        ),
        et_volume_veto_max_bias=float(
            args.et_volume_veto_max_bias
            if args.et_volume_veto_max_bias is not None
            else config.get("et_volume_veto_max_bias", 4.0)
        ),
        context_proxy_dim=int(proxy_dims.get("context_proxy_dim", 0)),
        disease_proxy_dim=int(proxy_dims.get("disease_proxy_dim", 0)),
        annotation_proxy_dim=int(proxy_dims.get("annotation_proxy_dim", 0)),
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
    should_adjust = _should_build_context_bank(args)
    bank_dataset = None
    if should_adjust:
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
    region_thresholds = parse_region_thresholds(args.region_thresholds)
    plausibility_base_thresholds = parse_region_thresholds(args.plausibility_region_base_thresholds)
    plausibility_low_thresholds = parse_region_thresholds(args.plausibility_region_low_thresholds)
    if (plausibility_base_thresholds is None) != (plausibility_low_thresholds is None):
        raise ValueError("--plausibility-region-base-thresholds and --plausibility-region-low-thresholds must be supplied together.")
    plausibility_region_thresholds = None
    if plausibility_base_thresholds is not None and plausibility_low_thresholds is not None:
        plausibility_region_thresholds = (
            plausibility_base_thresholds,
            plausibility_low_thresholds,
            float(args.plausibility_low_stability_wt_ratio_threshold),
            float(args.plausibility_low_stability_threshold),
            float(args.plausibility_tc_collapse_wt_ratio_min),
            float(args.plausibility_tc_collapse_wt_ratio_max),
            float(args.plausibility_tc_collapse_tc_ratio_threshold),
        )
    calibration_candidates = parse_threshold_candidates(args.calibration_thresholds)
    calibration_sweep = (
        BratsRegionThresholdSweep(calibration_candidates, objective=args.calibration_objective)
        if calibration_candidates
        else None
    )
    adjusted_calibration_sweep = (
        BratsRegionThresholdSweep(calibration_candidates, objective=args.calibration_objective)
        if calibration_candidates
        else None
    )
    mirror_axes = parse_mirror_tta_axes(args.mirror_tta_axes)
    context_bank = None
    if bank_dataset is not None:
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
    factual_region_calibrated_metrics: list[dict[str, float]] = []
    factual_plausibility_region_calibrated_metrics: list[dict[str, float]] = []
    factual_structural_metrics: list[dict[str, float]] = []
    factual_structural_region_calibrated_metrics: list[dict[str, float]] = []
    adjusted_batch_metrics: list[dict[str, float]] = []
    adjusted_region_calibrated_metrics: list[dict[str, float]] = []
    adjusted_plausibility_region_calibrated_metrics: list[dict[str, float]] = []
    adjusted_structural_metrics: list[dict[str, float]] = []
    adjusted_structural_region_calibrated_metrics: list[dict[str, float]] = []
    cct_consensus_batch_metrics: list[dict[str, float]] = []
    cct_stability_gated_batch_metrics: list[dict[str, float]] = []
    cct_consensus_region_calibrated_metrics: list[dict[str, float]] = []
    cct_stability_gated_region_calibrated_metrics: list[dict[str, float]] = []
    region_causal_batch_metrics: list[dict[str, float]] = []
    veto_metric_items: list[dict[str, float]] = []
    factual_volume_metrics: list[dict[str, float]] = []
    adjusted_volume_metrics: list[dict[str, float]] = []
    cct_consensus_volume_metrics: list[dict[str, float]] = []
    cct_stability_gated_volume_metrics: list[dict[str, float]] = []
    cct_instability_items: list[dict[str, float]] = []
    context_shifts: list[float] = []
    nearest_context_distances: list[float] = []
    case_records: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"mednext-brats-causal-eval:{args.split_name}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        cct_consensus_logits: Tensor | None = None
        cct_stability_gated_logits: Tensor | None = None
        outputs = model(
            image,
            context_bank=bank_device,
            max_adjustment_contexts=args.adjustment_contexts,
            adjustment_context_selection=args.adjustment_context_selection,
        )
        logits = outputs["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("CausalMedNeXt output 'logits' must be a tensor.")
        logits = mirror_tta_logits(
                lambda augmented: model(
                    augmented,
                    context_bank=bank_device,
                    max_adjustment_contexts=args.adjustment_contexts,
                    adjustment_context_selection=args.adjustment_context_selection,
                )["logits"],
            image,
            mirror_axes,
            base_logits=logits,
        )
        factual_losses.append(float(_segmentation_loss(logits, target).detach().cpu()))
        logits_cpu = logits.detach().cpu()
        target_cpu = target.detach().cpu()
        factual_batch_metrics.append(brats_region_metrics(logits_cpu, target_cpu, threshold=args.threshold))
        if region_thresholds is not None:
            factual_region_calibrated_metrics.append(brats_region_metrics_from_thresholds(logits_cpu, target_cpu, region_thresholds))
        if plausibility_region_thresholds is not None:
            (
                base_thresholds,
                low_thresholds,
                low_stability_wt_ratio_threshold,
                low_stability_threshold,
                tc_collapse_wt_ratio_min,
                tc_collapse_wt_ratio_max,
                tc_collapse_tc_ratio_threshold,
            ) = plausibility_region_thresholds
            factual_plausibility_region_calibrated_metrics.append(
                brats_region_metrics_from_plausibility_thresholds(
                    logits_cpu,
                    target_cpu,
                    base_thresholds,
                    low_thresholds,
                    low_stability_wt_ratio_threshold=low_stability_wt_ratio_threshold,
                    low_stability_threshold=low_stability_threshold,
                    tc_collapse_wt_ratio_min=tc_collapse_wt_ratio_min,
                    tc_collapse_wt_ratio_max=tc_collapse_wt_ratio_max,
                    tc_collapse_tc_ratio_threshold=tc_collapse_tc_ratio_threshold,
                )
            )
        if calibration_sweep is not None:
            calibration_sweep.update(logits_cpu, target_cpu)
        if args.structural_prior:
            factual_structural_metrics.append(
                brats_structural_region_metrics(
                    logits_cpu,
                    target_cpu,
                    threshold=args.structural_threshold,
                    min_component_size=args.structural_min_component_size,
                    fill_holes=args.structural_fill_holes,
                    keep_largest=args.structural_keep_largest,
                )
            )
            if region_thresholds is not None:
                factual_structural_region_calibrated_metrics.append(
                    brats_structural_region_metrics_from_thresholds(
                        logits_cpu,
                        target_cpu,
                        region_thresholds,
                        min_component_size=args.structural_min_component_size,
                        fill_holes=args.structural_fill_holes,
                        keep_largest=args.structural_keep_largest,
                    )
                )
        veto_metric_items.append(_et_volume_veto_metric_item(outputs))
        factual_volume_metrics.extend(_volume_metrics(logits, target, args.threshold))
        region_causal_logits = outputs.get("region_causal_logits")
        if isinstance(region_causal_logits, Tensor):
            region_causal_logits = mirror_tta_logits(
                lambda augmented: model(
                    augmented,
                    context_bank=bank_device,
                    max_adjustment_contexts=args.adjustment_contexts,
                    adjustment_context_selection=args.adjustment_context_selection,
                )["region_causal_logits"],
                image,
                mirror_axes,
                base_logits=region_causal_logits,
            )
            region_causal_batch_metrics.append(
                brats_region_metrics(region_causal_logits.detach().cpu(), target.detach().cpu(), threshold=args.threshold)
            )

        z_c = outputs.get("z_c")
        if isinstance(z_c, Tensor):
            nearest_context_distances.extend(_context_overlap(z_c, bank_device).get("overlap/nearest_context_l2", []))

        if int(args.cct_contexts) > 0:
            if bank_device is None:
                raise ValueError("--cct-contexts requires a non-empty context bank.")
            features = outputs.get("features")
            z_d = outputs.get("z_d")
            z_c = outputs.get("z_c")
            if not isinstance(features, tuple) or not all(isinstance(feature, Tensor) for feature in features):
                raise TypeError("CausalMedNeXt CCT needs tuple Tensor output 'features'.")
            if not isinstance(z_d, Tensor):
                raise TypeError("CausalMedNeXt CCT needs Tensor output 'z_d'.")
            if not isinstance(z_c, Tensor):
                raise TypeError("CausalMedNeXt CCT needs Tensor output 'z_c'.")
            cct_outputs = _cct_transport_outputs(
                model,
                features,
                z_d,
                z_c,
                bank_device,
                image,
                max_contexts=args.cct_contexts,
                selection=args.cct_selection,
                instability_scale=args.cct_instability_scale,
            )
            cct_consensus_logits = cct_outputs["consensus_logits"]
            cct_stability_gated_logits = cct_outputs["stability_gated_logits"]
            cct_consensus_cpu = cct_consensus_logits.detach().cpu()
            cct_stability_gated_cpu = cct_stability_gated_logits.detach().cpu()
            cct_consensus_batch_metrics.append(brats_region_metrics(cct_consensus_cpu, target_cpu, threshold=args.threshold))
            cct_stability_gated_batch_metrics.append(
                brats_region_metrics(cct_stability_gated_cpu, target_cpu, threshold=args.threshold)
            )
            if region_thresholds is not None:
                cct_consensus_region_calibrated_metrics.append(
                    brats_region_metrics_from_thresholds(cct_consensus_cpu, target_cpu, region_thresholds)
                )
                cct_stability_gated_region_calibrated_metrics.append(
                    brats_region_metrics_from_thresholds(cct_stability_gated_cpu, target_cpu, region_thresholds)
                )
            cct_consensus_volume_metrics.extend(_volume_metrics(cct_consensus_logits, target, args.threshold))
            cct_stability_gated_volume_metrics.extend(_volume_metrics(cct_stability_gated_logits, target, args.threshold))
            instability = cct_outputs["instability"].detach()
            positive_instability = instability[torch.sigmoid(cct_consensus_logits.detach()) >= float(args.threshold)]
            cct_instability_items.append(
                {
                    "cct/transported_context_count": float(cct_outputs["transported_context_count"].detach().cpu()),
                    "cct/instability_mean": float(instability.mean().cpu()),
                    "cct/instability_max": float(instability.max().cpu()),
                    "cct/positive_instability_mean": float(positive_instability.mean().cpu())
                    if positive_instability.numel() > 0
                    else 0.0,
                    "cct/positive_instability_fraction": float(
                        (positive_instability > float(args.cct_instability_threshold)).float().mean().cpu()
                    )
                    if positive_instability.numel() > 0
                    else 0.0,
                }
            )

        adjusted = outputs.get("adjusted_logits")
        if isinstance(adjusted, Tensor):
            adjusted = mirror_tta_logits(
                lambda augmented: model(
                    augmented,
                    context_bank=bank_device,
                    max_adjustment_contexts=args.adjustment_contexts,
                    adjustment_context_selection=args.adjustment_context_selection,
                )["adjusted_logits"],
                image,
                mirror_axes,
                base_logits=adjusted,
            )
            adjusted_losses.append(float(_segmentation_loss(adjusted, target).detach().cpu()))
            adjusted_cpu = adjusted.detach().cpu()
            adjusted_batch_metrics.append(brats_region_metrics(adjusted_cpu, target_cpu, threshold=args.threshold))
            if region_thresholds is not None:
                adjusted_region_calibrated_metrics.append(brats_region_metrics_from_thresholds(adjusted_cpu, target_cpu, region_thresholds))
            if plausibility_region_thresholds is not None:
                (
                    base_thresholds,
                    low_thresholds,
                    low_stability_wt_ratio_threshold,
                    low_stability_threshold,
                    tc_collapse_wt_ratio_min,
                    tc_collapse_wt_ratio_max,
                    tc_collapse_tc_ratio_threshold,
                ) = plausibility_region_thresholds
                adjusted_plausibility_region_calibrated_metrics.append(
                    brats_region_metrics_from_plausibility_thresholds(
                        adjusted_cpu,
                        target_cpu,
                        base_thresholds,
                        low_thresholds,
                        low_stability_wt_ratio_threshold=low_stability_wt_ratio_threshold,
                        low_stability_threshold=low_stability_threshold,
                        tc_collapse_wt_ratio_min=tc_collapse_wt_ratio_min,
                        tc_collapse_wt_ratio_max=tc_collapse_wt_ratio_max,
                        tc_collapse_tc_ratio_threshold=tc_collapse_tc_ratio_threshold,
                    )
                )
            if adjusted_calibration_sweep is not None:
                adjusted_calibration_sweep.update(adjusted_cpu, target_cpu)
            if args.structural_prior:
                adjusted_structural_metrics.append(
                    brats_structural_region_metrics(
                        adjusted_cpu,
                        target_cpu,
                        threshold=args.structural_threshold,
                        min_component_size=args.structural_min_component_size,
                        fill_holes=args.structural_fill_holes,
                        keep_largest=args.structural_keep_largest,
                    )
                )
                if region_thresholds is not None:
                    adjusted_structural_region_calibrated_metrics.append(
                        brats_structural_region_metrics_from_thresholds(
                            adjusted_cpu,
                            target_cpu,
                            region_thresholds,
                            min_component_size=args.structural_min_component_size,
                            fill_holes=args.structural_fill_holes,
                            keep_largest=args.structural_keep_largest,
                        )
                    )
            adjusted_volume_metrics.extend(_volume_metrics(adjusted, target, args.threshold))
            context_shifts.append(float((torch.sigmoid(logits) - torch.sigmoid(adjusted)).abs().mean().detach().cpu()))

        if args.include_per_case:
            variants: dict[str, Tensor] = {"factual": logits}
            if isinstance(region_causal_logits, Tensor):
                variants["region_causal"] = region_causal_logits
            if isinstance(adjusted, Tensor):
                variants["adjusted"] = adjusted
            if isinstance(cct_consensus_logits, Tensor):
                variants["cct_consensus"] = cct_consensus_logits
            if isinstance(cct_stability_gated_logits, Tensor):
                variants["cct_stability_gated"] = cct_stability_gated_logits
            for item_index in range(image.shape[0]):
                case_records.append(
                    _brats_case_record(
                        batch,
                        item_index,
                        target,
                        args.threshold,
                        variants,
                        region_thresholds=region_thresholds,
                        plausibility_region_thresholds=plausibility_region_thresholds,
                    )
                )

        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    scm = default_utsw_scm()
    metrics: dict[str, Any] = {
        "method": "Causal MedNeXt on BraTS H5",
        "causal_question": scm.question.query,
        "causal_estimand": scm.question.estimand,
        "causal_warning": (
            "BraTS H5 evaluation has no clinical metadata proxy columns; checkpoint proxy_dims indicate whether "
            "dataset-agnostic pseudo-proxies were used during training."
        ),
        "proxy_dims": checkpoint.get("proxy_dims", {}),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split_name,
        "brats_csv": str(args.brats_csv),
        "context_csv": str(args.context_csv or args.brats_csv),
        "threshold": float(args.threshold),
        "mirror_tta_axes": ",".join(str(axis) for axis in mirror_axes),
        "num_cases": float(len(factual_volume_metrics)),
        "context_bank_size": float(0 if context_bank is None else context_bank.shape[0]),
        "adjustment_context_selection": str(args.adjustment_context_selection),
        "cct_contexts": int(args.cct_contexts),
        "cct_selection": str(args.cct_selection),
        "cct_instability_scale": float(args.cct_instability_scale),
        "cct_instability_threshold": float(args.cct_instability_threshold),
        "structural_prior": bool(args.structural_prior),
        "structural_threshold": float(args.structural_threshold),
        "structural_min_component_size": int(args.structural_min_component_size),
        "structural_fill_holes": bool(args.structural_fill_holes),
        "structural_keep_largest": bool(args.structural_keep_largest),
        "plausibility_region_base_thresholds": str(args.plausibility_region_base_thresholds or ""),
        "plausibility_region_low_thresholds": str(args.plausibility_region_low_thresholds or ""),
        "plausibility_low_stability_wt_ratio_threshold": float(args.plausibility_low_stability_wt_ratio_threshold),
        "plausibility_low_stability_threshold": float(args.plausibility_low_stability_threshold),
        "plausibility_tc_collapse_wt_ratio_min": float(args.plausibility_tc_collapse_wt_ratio_min),
        "plausibility_tc_collapse_wt_ratio_max": float(args.plausibility_tc_collapse_wt_ratio_max),
        "plausibility_tc_collapse_tc_ratio_threshold": float(args.plausibility_tc_collapse_tc_ratio_threshold),
        "factual/loss": _mean(factual_losses),
    }
    metrics.update(_average_metric_dicts(factual_batch_metrics))
    if factual_region_calibrated_metrics:
        metrics.update(prefix_metrics(_average_metric_dicts(factual_region_calibrated_metrics), "region_calibrated"))
    if factual_plausibility_region_calibrated_metrics:
        metrics.update(
            prefix_metrics(
                _average_metric_dicts(factual_plausibility_region_calibrated_metrics),
                "plausibility_region_calibrated",
            )
        )
    if calibration_sweep is not None:
        metrics.update(prefix_metrics(calibration_sweep.summary(), "sweep_region_calibrated"))
    if factual_structural_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(factual_structural_metrics), "structural"))
    if factual_structural_region_calibrated_metrics:
        metrics.update(
            _prefix_metrics(
                _average_metric_dicts(factual_structural_region_calibrated_metrics),
                "structural_region_calibrated",
            )
        )
    if region_causal_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(region_causal_batch_metrics), "region_causal"))
    metrics.update(_average_metric_dicts(veto_metric_items))
    metrics.update({f"volume/{key}": value for key, value in _average_metric_dicts(factual_volume_metrics).items()})
    if adjusted_losses:
        metrics["adjusted/loss"] = _mean(adjusted_losses)
        metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_batch_metrics), "adjusted"))
        if adjusted_region_calibrated_metrics:
            metrics.update(prefix_metrics(_average_metric_dicts(adjusted_region_calibrated_metrics), "adjusted_region_calibrated"))
        if adjusted_plausibility_region_calibrated_metrics:
            metrics.update(
                prefix_metrics(
                    _average_metric_dicts(adjusted_plausibility_region_calibrated_metrics),
                    "adjusted_plausibility_region_calibrated",
                )
            )
        if adjusted_calibration_sweep is not None:
            metrics.update(prefix_metrics(adjusted_calibration_sweep.summary(), "adjusted_sweep_region_calibrated"))
        if adjusted_structural_metrics:
            metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_structural_metrics), "adjusted_structural"))
        if adjusted_structural_region_calibrated_metrics:
            metrics.update(
                _prefix_metrics(
                    _average_metric_dicts(adjusted_structural_region_calibrated_metrics),
                    "adjusted_structural_region_calibrated",
                )
            )
        metrics.update({f"adjusted/volume/{key}": value for key, value in _average_metric_dicts(adjusted_volume_metrics).items()})
        metrics["intervention/context_adjustment_mean_abs_prob_shift"] = _mean(context_shifts)
        metrics["intervention/adjusted_minus_factual_mean_dice"] = float(metrics.get("adjusted/brats/mean_dice", float("nan"))) - float(metrics.get("brats/mean_dice", float("nan")))
    if cct_consensus_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(cct_consensus_batch_metrics), "cct_consensus"))
        metrics.update(_prefix_metrics(_average_metric_dicts(cct_stability_gated_batch_metrics), "cct_stability_gated"))
        if cct_consensus_region_calibrated_metrics:
            metrics.update(
                prefix_metrics(
                    _average_metric_dicts(cct_consensus_region_calibrated_metrics),
                    "cct_consensus_region_calibrated",
                )
            )
        if cct_stability_gated_region_calibrated_metrics:
            metrics.update(
                prefix_metrics(
                    _average_metric_dicts(cct_stability_gated_region_calibrated_metrics),
                    "cct_stability_gated_region_calibrated",
                )
            )
        metrics.update({f"cct_consensus/volume/{key}": value for key, value in _average_metric_dicts(cct_consensus_volume_metrics).items()})
        metrics.update(
            {
                f"cct_stability_gated/volume/{key}": value
                for key, value in _average_metric_dicts(cct_stability_gated_volume_metrics).items()
            }
        )
        metrics.update(_average_metric_dicts(cct_instability_items))
        metrics["intervention/cct_consensus_minus_factual_mean_dice"] = float(
            metrics.get("cct_consensus/brats/mean_dice", float("nan"))
        ) - float(metrics.get("brats/mean_dice", float("nan")))
        metrics["intervention/cct_stability_gated_minus_factual_mean_dice"] = float(
            metrics.get("cct_stability_gated/brats/mean_dice", float("nan"))
        ) - float(metrics.get("brats/mean_dice", float("nan")))
    if case_records:
        metrics["per_case"] = case_records
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
    parser.add_argument("--region-volume-scale", type=float)
    parser.add_argument("--et-volume-veto-scale", type=float)
    parser.add_argument("--et-volume-veto-multiplier", type=float)
    parser.add_argument("--et-volume-veto-min-fraction", type=float)
    parser.add_argument("--et-volume-veto-max-bias", type=float)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--region-thresholds", help="Fixed WT/TC/ET thresholds, e.g. 'WT=0.4,TC=0.5,ET=0.6'.")
    parser.add_argument("--structural-prior", action="store_true")
    parser.add_argument("--structural-threshold", type=float, default=0.5)
    parser.add_argument("--structural-min-component-size", type=int, default=16)
    parser.add_argument("--structural-fill-holes", action="store_true")
    parser.add_argument("--structural-keep-largest", action="store_true")
    parser.add_argument("--plausibility-region-base-thresholds", help="Base WT/TC/ET thresholds for plausibility-gated calibration.")
    parser.add_argument("--plausibility-region-low-thresholds", help="Lower WT/TC/ET thresholds for plausibility-gated calibration.")
    parser.add_argument("--plausibility-low-stability-wt-ratio-threshold", type=float, default=0.0)
    parser.add_argument("--plausibility-low-stability-threshold", type=float, default=0.0)
    parser.add_argument("--plausibility-tc-collapse-wt-ratio-min", type=float, default=0.0)
    parser.add_argument("--plausibility-tc-collapse-wt-ratio-max", type=float, default=0.0)
    parser.add_argument("--plausibility-tc-collapse-tc-ratio-threshold", type=float, default=0.0)
    parser.add_argument("--calibration-thresholds", help="Comma-separated WT/TC/ET threshold grid for validation-time sweep.")
    parser.add_argument(
        "--calibration-objective",
        choices=CALIBRATION_OBJECTIVES,
        default="mean",
        help="Objective used when choosing thresholds from --calibration-thresholds.",
    )
    parser.add_argument("--mirror-tta-axes", help="Optional spatial mirror TTA axes: d,h,w or z,y,x.")
    parser.add_argument("--include-per-case", action="store_true", help="Write per-volume Dice and calibration diagnostics.")
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument(
        "--adjustment-context-selection",
        choices=["uniform", "nearest", "farthest", "diverse-nearest"],
        default="uniform",
        help="How the SCM adjusted logits select proxy contexts for each target case.",
    )
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument(
        "--cct-contexts",
        type=int,
        default=0,
        help="Evaluate Counterfactual Context Transport consensus using this many proxy contexts.",
    )
    parser.add_argument(
        "--cct-selection",
        choices=["uniform", "nearest", "farthest", "diverse-nearest"],
        default="uniform",
        help=(
            "How to choose proxy contexts for CCT. 'uniform' estimates a context marginal, "
            "'nearest' adapts to the anchor context, 'farthest' stress-tests shift, and "
            "'diverse-nearest' balances support-awareness with diversity."
        ),
    )
    parser.add_argument(
        "--cct-instability-scale",
        type=float,
        default=1.0,
        help="Lower-confidence-bound scale for CCT stability-gated logits.",
    )
    parser.add_argument(
        "--cct-instability-threshold",
        type=float,
        default=0.05,
        help="Report fraction of positive CCT voxels above this context-instability value.",
    )
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
