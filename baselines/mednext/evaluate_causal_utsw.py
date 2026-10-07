from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - scipy is expected, but keep CLI robust.
    ndi = None

from baselines.mednext.calibration import (
    CALIBRATION_OBJECTIVES,
    BratsAdaptiveRegionThresholdSweep,
    BratsRegionThresholdSweep,
    brats_region_probabilities,
    brats_region_metrics_from_adaptive_thresholds,
    brats_region_metrics_from_plausibility_thresholds,
    brats_region_metrics_from_thresholds,
    parse_fraction_candidates,
    parse_region_thresholds,
    parse_threshold_candidates,
    prefix_metrics,
)
from baselines.mednext.causal import CausalMedNeXt, build_causal_mednext
from baselines.mednext.common import mirror_tta_logits, parse_mirror_tta_axes, registered_modality_consistency_metrics
from baselines.mednext.train_causal_utsw import _et_volume_veto_metric_item, apply_style_intervention
from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.data import UTSWGliomaDataset, UTSW_MODALITIES
from baselines.segformer3d.evaluate_causal_utsw import _context_overlap, _proxy_layout, _proxy_losses, _segmentation_loss, _volume_metrics
from baselines.segformer3d.train_causal_utsw import build_context_bank, _metadata_dims, _prefix_metrics
from baselines.segformer3d.train_utsw import _average_metric_dicts, _mean, _resolve_device
from crn.metrics import brats_region_metrics, brats_region_metrics_from_region_logits, brats_structural_region_metrics


def _logit(probability: Tensor) -> Tensor:
    probability = probability.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(probability) - torch.log1p(-probability)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_ids_from_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    case_ids = [item.strip() for item in value.split(",") if item.strip()]
    return case_ids or None


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _config_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    return config.get(name, default) if value is None else value


def _make_dataset(root: Path, case_ids: list[str], args: argparse.Namespace, config: dict[str, Any]) -> UTSWGliomaDataset:
    return UTSWGliomaDataset(
        root=root,
        volume_size=int(_config_value(args, config, "volume_size", 64)),
        case_ids=case_ids,
        crop_margin=int(_config_value(args, config, "crop_margin", 8)),
        prefer_manual_seg=bool(_config_value(args, config, "prefer_manual_seg", False)),
        use_ants_modalities=bool(_config_value(args, config, "use_ants_modalities", False)),
        metadata_path=_config_value(args, config, "metadata_path"),
        include_metadata=True,
    )


def _make_dataset_with_modality_source(
    root: Path,
    case_ids: list[str],
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    use_ants_modalities: bool,
) -> UTSWGliomaDataset:
    return UTSWGliomaDataset(
        root=root,
        volume_size=int(_config_value(args, config, "volume_size", 64)),
        case_ids=case_ids,
        crop_margin=int(_config_value(args, config, "crop_margin", 8)),
        prefer_manual_seg=bool(_config_value(args, config, "prefer_manual_seg", False)),
        use_ants_modalities=use_ants_modalities,
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


def _build_model(checkpoint: dict[str, Any], dataset: UTSWGliomaDataset, args: argparse.Namespace, config: dict[str, Any]) -> CausalMedNeXt:
    model = build_causal_mednext(
        model_id=str(_config_value(args, config, "model_id", "S")),
        kernel_size=int(_config_value(args, config, "kernel_size", 3)),
        latent_dim=int(_config_value(args, config, "latent_dim", 128)),
        num_classes=3,
        base_channels=_config_value(args, config, "base_channels"),
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
        region_volume_scale=float(_config_value(args, config, "region_volume_scale", 1000.0)),
        et_volume_veto_scale=float(_config_value(args, config, "et_volume_veto_scale", 0.0)),
        et_volume_veto_multiplier=float(_config_value(args, config, "et_volume_veto_multiplier", 4.0)),
        et_volume_veto_min_fraction=float(_config_value(args, config, "et_volume_veto_min_fraction", 5e-4)),
        et_volume_veto_max_bias=float(_config_value(args, config, "et_volume_veto_max_bias", 4.0)),
        **_proxy_dims(checkpoint, dataset),
    )
    model.load_compatible_state_dict(checkpoint["model"])
    return model


def _fuse_style_tta_logits(
    factual_logits: Tensor,
    style_logits: Tensor,
    fusion: str,
    batch: dict[str, Any] | None = None,
) -> Tensor:
    if fusion == "enhancing-only":
        fused = factual_logits.clone()
        fused[:, 2:3] = style_logits[:, 2:3]
        return fused
    if fusion == "enhancing-confident":
        fused = factual_logits.clone()
        factual_et = factual_logits[:, 2:3]
        style_et = style_logits[:, 2:3]
        factual_confidence = (torch.sigmoid(factual_et) - 0.5).abs()
        style_confidence = (torch.sigmoid(style_et) - 0.5).abs()
        fused[:, 2:3] = torch.where(style_confidence > factual_confidence, style_et, factual_et)
        return fused
    if fusion == "enhancing-union":
        fused = factual_logits.clone()
        fused[:, 2:3] = torch.maximum(factual_logits[:, 2:3], style_logits[:, 2:3])
        return fused
    if fusion == "enhancing-intersection":
        fused = factual_logits.clone()
        fused[:, 2:3] = torch.minimum(factual_logits[:, 2:3], style_logits[:, 2:3])
        return fused
    if fusion == "enhancing-demote-core":
        factual_prob = torch.sigmoid(factual_logits)
        style_prob = torch.sigmoid(style_logits)
        kept_et = torch.minimum(factual_prob[:, 2:3], style_prob[:, 2:3])
        demoted_et = (factual_prob[:, 2:3] - kept_et).clamp(0.0, 1.0)
        fused_prob = factual_prob.clone()
        fused_prob[:, 2:3] = kept_et
        fused_prob[:, 0:1] = 1.0 - (1.0 - factual_prob[:, 0:1]) * (1.0 - demoted_et)
        return _logit(fused_prob)
    if fusion == "enhancing-empty-consensus-demote-core":
        factual_prob = torch.sigmoid(factual_logits)
        style_prob = torch.sigmoid(style_logits)
        kept_et = torch.minimum(factual_prob[:, 2:3], style_prob[:, 2:3])
        demoted_et = (factual_prob[:, 2:3] - kept_et).clamp(0.0, 1.0)
        demoted_prob = factual_prob.clone()
        demoted_prob[:, 2:3] = kept_et
        demoted_prob[:, 0:1] = 1.0 - (1.0 - factual_prob[:, 0:1]) * (1.0 - demoted_et)
        spatial_dims = tuple(range(1, factual_prob[:, 2].ndim))
        factual_et_voxels = (factual_prob[:, 2] > 0.5).sum(dim=spatial_dims)
        style_et_voxels = (style_prob[:, 2] > 0.5).sum(dim=spatial_dims)
        use_demotion = ((factual_et_voxels > 0) & (style_et_voxels == 0)).view(-1, 1, 1, 1, 1)
        fused_prob = torch.where(use_demotion, demoted_prob, factual_prob)
        return _logit(fused_prob)
    if fusion == "enhancing-component-consensus-demote-core":
        return _apply_component_consensus_et_demotion(factual_logits, style_logits)
    if fusion == "phenotype-enhancing-demote-core":
        return _apply_phenotype_gated_et_demotion(factual_logits, style_logits, batch)
    return style_logits


def _component_consensus_unstable_et_mask(
    factual_et_prob: Tensor,
    style_et_prob: Tensor,
    threshold: float = 0.5,
    min_style_overlap: float = 0.25,
    min_style_mean: float = 0.35,
) -> Tensor:
    """Find ET components contradicted by the style-intervention counterfactual."""
    if factual_et_prob.ndim != 4 or style_et_prob.shape != factual_et_prob.shape:
        raise ValueError(
            "Expected factual/style ET probabilities shaped (B, D, H, W), "
            f"got {tuple(factual_et_prob.shape)} and {tuple(style_et_prob.shape)}"
        )
    unstable = torch.zeros_like(factual_et_prob, dtype=torch.bool)
    factual_mask = factual_et_prob > float(threshold)
    style_mask = style_et_prob > float(threshold)
    for batch_index in range(factual_et_prob.shape[0]):
        mask = factual_mask[batch_index]
        if int(mask.sum().item()) <= 0:
            continue
        if ndi is None:
            overlap = (style_mask[batch_index] & mask).sum().to(dtype=torch.float32) / mask.sum().clamp_min(1)
            mean_style = style_et_prob[batch_index][mask].mean()
            if float(overlap.detach().cpu()) < min_style_overlap and float(mean_style.detach().cpu()) < min_style_mean:
                unstable[batch_index] = mask
            continue
        labels_cpu, component_count = ndi.label(mask.detach().cpu().numpy())
        labels = torch.from_numpy(labels_cpu).to(device=mask.device)
        for component_id in range(1, int(component_count) + 1):
            component = labels == component_id
            component_voxels = int(component.sum().item())
            if component_voxels <= 0:
                continue
            style_overlap = (style_mask[batch_index] & component).sum().to(dtype=torch.float32) / float(component_voxels)
            mean_style = style_et_prob[batch_index][component].mean()
            if float(style_overlap.detach().cpu()) < min_style_overlap and float(mean_style.detach().cpu()) < min_style_mean:
                unstable[batch_index] |= component
    return unstable


def _apply_component_consensus_et_demotion(factual_logits: Tensor, style_logits: Tensor) -> Tensor:
    """Demote only ET components that disappear under do(style = s')."""
    factual_prob = torch.sigmoid(factual_logits)
    style_prob = torch.sigmoid(style_logits)
    unstable = _component_consensus_unstable_et_mask(factual_prob[:, 2], style_prob[:, 2]).unsqueeze(1)
    if not bool(unstable.any().detach().cpu()):
        return factual_logits
    kept_et = torch.where(unstable, torch.minimum(factual_prob[:, 2:3], style_prob[:, 2:3]), factual_prob[:, 2:3])
    demoted_et = (factual_prob[:, 2:3] - kept_et).clamp(0.0, 1.0)
    fused_prob = factual_prob.clone()
    fused_prob[:, 2:3] = kept_et
    fused_prob[:, 0:1] = 1.0 - (1.0 - factual_prob[:, 0:1]) * (1.0 - demoted_et)
    return _logit(fused_prob)


def _registered_modality_stability_score(
    native_logits: Tensor,
    registered_logits: Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    native_regions = brats_region_probabilities(native_logits.detach().float())
    registered_regions = brats_region_probabilities(registered_logits.detach().float())
    native_pred = (native_regions >= float(threshold)).float()
    registered_pred = (registered_regions >= float(threshold)).float()
    dims = tuple(range(2, native_pred.ndim))
    intersection = (native_pred * registered_pred).sum(dim=dims)
    denominator = native_pred.sum(dim=dims) + registered_pred.sum(dim=dims)
    agreement = (2.0 * intersection + eps) / (denominator + eps)
    region_similarity = 1.0 - (native_regions - registered_regions).abs().mean(dim=dims)
    return 0.5 * (agreement.mean(dim=1) + region_similarity.mean(dim=1))


def _registered_modality_stability_gate_mask(
    native_logits: Tensor,
    registered_logits: Tensor,
    *,
    stability_gate_threshold: float = 0.9,
) -> tuple[Tensor, Tensor]:
    stability = _registered_modality_stability_score(native_logits, registered_logits)
    return stability, stability < float(stability_gate_threshold)


def _fuse_registered_modality_logits(
    native_logits: Tensor,
    registered_logits: Tensor,
    fusion: str,
    *,
    stability_gate_threshold: float = 0.9,
) -> Tensor:
    if fusion == "mean-logits":
        return 0.5 * (native_logits + registered_logits)
    native_prob = torch.sigmoid(native_logits)
    registered_prob = torch.sigmoid(registered_logits)
    if fusion == "mean-probs":
        return _logit(0.5 * (native_prob + registered_prob))
    if fusion == "max-probs":
        return _logit(torch.maximum(native_prob, registered_prob))
    if fusion == "registered-only":
        return registered_logits
    if fusion == "stability-gated-registered":
        _, gate_mask = _registered_modality_stability_gate_mask(
            native_logits,
            registered_logits,
            stability_gate_threshold=stability_gate_threshold,
        )
        gate = gate_mask.view(-1, *([1] * (native_prob.ndim - 1)))
        fused_prob = torch.where(gate, registered_prob, 0.5 * (native_prob + registered_prob))
        return _logit(fused_prob)
    raise ValueError(f"Unsupported registered modality fusion: {fusion}")


def _apply_phenotype_gated_et_demotion(
    factual_logits: Tensor,
    style_logits: Tensor,
    batch: dict[str, Any] | None,
) -> Tensor:
    """Use observed phenotype as a causal gate for ET -> NCR/NET demotion."""
    if batch is None:
        return factual_logits
    factual_prob = torch.sigmoid(factual_logits)
    style_prob = torch.sigmoid(style_logits)
    demoted_prob = factual_prob.clone()
    kept_et = torch.minimum(factual_prob[:, 2:3], style_prob[:, 2:3])
    demoted_et = (factual_prob[:, 2:3] - kept_et).clamp(0.0, 1.0)
    demoted_prob[:, 2:3] = kept_et
    demoted_prob[:, 0:1] = 1.0 - (1.0 - factual_prob[:, 0:1]) * (1.0 - demoted_et)
    use_demotion = [
        _metadata_supports_nonenhancing_core(batch, batch_index)
        for batch_index in range(factual_logits.shape[0])
    ]
    gate = torch.as_tensor(use_demotion, device=factual_logits.device, dtype=torch.bool).view(-1, 1, 1, 1, 1)
    if not bool(gate.any().detach().cpu()):
        return factual_logits
    fused_prob = torch.where(gate, demoted_prob, factual_prob)
    return _logit(fused_prob)


def _metadata_raw_value(batch: dict[str, Any], column: str, batch_index: int) -> Any:
    raw = batch.get("metadata_raw")
    if not isinstance(raw, dict):
        return None
    value = raw.get(column)
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.item()
        return value[batch_index].item()
    if isinstance(value, (list, tuple)):
        if batch_index >= len(value):
            return None
        return value[batch_index]
    return value


def _metadata_text(batch: dict[str, Any], column: str, batch_index: int) -> str:
    value = _metadata_raw_value(batch, column, batch_index)
    if value is None:
        return ""
    return str(value).strip().lower()


def _metadata_float(batch: dict[str, Any], column: str, batch_index: int) -> float | None:
    text = _metadata_text(batch, column, batch_index)
    if not text or text in {"na", "n/a", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _metadata_supports_nonenhancing_core(batch: dict[str, Any], batch_index: int) -> bool:
    """Gate the intervention to lower-grade/non-GBM phenotypes when metadata exists."""
    tumor_type = _metadata_text(batch, "Tumor Type", batch_index)
    grade = _metadata_float(batch, "Tumor Grade", batch_index)
    if not tumor_type and grade is None:
        return False
    if "glioblastoma" in tumor_type or "gbm" in tumor_type:
        return False
    if grade is not None and grade <= 3.0:
        return True
    lower_grade_markers = ("astrocytoma", "oligodendroglioma", "oligoastrocytoma", "other")
    return any(marker in tumor_type for marker in lower_grade_markers)


def _central_wt_core_mask(wt_mask: Tensor, fraction: float) -> Tensor:
    """Select the most central WT voxels as a non-enhancing-core counterfactual."""
    if wt_mask.ndim != 3:
        raise ValueError(f"Expected a 3D WT mask, got shape {tuple(wt_mask.shape)}")
    wt_mask = wt_mask.bool()
    wt_voxels = int(wt_mask.sum().item())
    if wt_voxels <= 0:
        return torch.zeros_like(wt_mask)
    fraction = max(0.0, min(1.0, float(fraction)))
    if fraction <= 0.0:
        return torch.zeros_like(wt_mask)
    keep_voxels = max(1, min(wt_voxels, int(round(wt_voxels * fraction))))
    if ndi is not None:
        distances_cpu = ndi.distance_transform_edt(wt_mask.detach().cpu().numpy()).astype("float32")
        scores = torch.from_numpy(distances_cpu).to(device=wt_mask.device)
    else:
        coords = torch.nonzero(wt_mask, as_tuple=False).float()
        center = coords.mean(dim=0)
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(wt_mask.shape[0], device=wt_mask.device),
                torch.arange(wt_mask.shape[1], device=wt_mask.device),
                torch.arange(wt_mask.shape[2], device=wt_mask.device),
                indexing="ij",
            ),
            dim=-1,
        ).float()
        scores = -(grid - center).square().sum(dim=-1).sqrt()
    flat_wt = wt_mask.flatten()
    candidate_indices = flat_wt.nonzero(as_tuple=False).flatten()
    candidate_scores = scores.flatten()[candidate_indices]
    chosen = torch.topk(candidate_scores, k=keep_voxels, largest=True).indices
    selected_indices = candidate_indices[chosen]
    out = torch.zeros_like(flat_wt)
    out[selected_indices] = True
    return out.view_as(wt_mask)


def _apply_nonenhancing_core_completion(
    logits: Tensor,
    batch: dict[str, Any],
    *,
    threshold: float,
    fraction: float,
    min_wt_voxels: int,
    max_tc_voxels: int,
    metadata_gate: bool,
) -> tuple[Tensor, list[bool]]:
    """Estimate P(Y_core | do(non-enhancing-core present), WT mediator) for a missed-core phenotype.

    The operation is deliberately narrow: it can only add NCR/NET voxels inside an already predicted
    whole-tumor region, and it never adds edema or enhancing tumor. This makes the post-intervention
    answer interpretable as non-enhancing-core completion instead of generic mask smoothing.
    """
    if logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError(f"Expected logits shaped (B, 3, D, H, W), got {tuple(logits.shape)}")
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= float(threshold)
    updated = probabilities.clone()
    triggered: list[bool] = []
    for batch_index in range(logits.shape[0]):
        if metadata_gate and not _metadata_supports_nonenhancing_core(batch, batch_index):
            triggered.append(False)
            continue
        ncr = predicted[batch_index, 0]
        edema = predicted[batch_index, 1]
        enhancing = predicted[batch_index, 2]
        wt = ncr | edema | enhancing
        tc = ncr | enhancing
        wt_voxels = int(wt.sum().item())
        tc_voxels = int(tc.sum().item())
        enhancing_voxels = int(enhancing.sum().item())
        should_complete = (
            enhancing_voxels == 0
            and tc_voxels <= int(max_tc_voxels)
            and wt_voxels >= int(min_wt_voxels)
        )
        if not should_complete:
            triggered.append(False)
            continue
        core_mask = _central_wt_core_mask(wt, fraction=float(fraction))
        if int(core_mask.sum().item()) <= 0:
            triggered.append(False)
            continue
        updated[batch_index, 0] = torch.maximum(
            updated[batch_index, 0],
            core_mask.to(dtype=updated.dtype, device=updated.device),
        )
        triggered.append(True)
    if not any(triggered):
        return logits, triggered
    return _logit(updated), triggered


def _apply_nested_region_consistency(logits: Tensor) -> Tensor:
    """Map subregion logits through a valid WT -> TC -> ET nested-region SCM."""
    if logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError(f"Expected logits shaped (B, 3, D, H, W), got {tuple(logits.shape)}")
    region_prob = CausalMedNeXt._subregion_prob_to_region_prob(torch.sigmoid(logits))
    raw_region_logits = _logit(region_prob)
    zeros = torch.zeros_like(raw_region_logits)
    _, _, nested_subregion_logits = CausalMedNeXt._nested_condition_logits_to_outputs(raw_region_logits, zeros)
    return nested_subregion_logits


def _batch_case_ids(batch: dict[str, Any], batch_size: int) -> list[str]:
    case_ids = batch.get("case_id")
    if isinstance(case_ids, Tensor):
        return [str(value.item()) for value in case_ids]
    if isinstance(case_ids, (list, tuple)):
        return [str(value) for value in case_ids]
    if case_ids is None:
        return [f"batch_item_{index}" for index in range(batch_size)]
    if batch_size == 1:
        return [str(case_ids)]
    return [f"{case_ids}_{index}" for index in range(batch_size)]


def _stable_style_seed(base_seed: int, sample_index: int, case_ids: list[str]) -> int:
    """Make style interventions reproducible for a case/sample, independent of loader order."""
    key = f"{int(base_seed)}:{int(sample_index)}:{'|'.join(case_ids)}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)


def _style_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [0 if device.index is None else int(device.index)]


def _case_metrics(logits: Tensor, target: Tensor, threshold: float) -> dict[str, float]:
    return brats_region_metrics(
        logits.detach().cpu(),
        target.detach().cpu(),
        threshold=threshold,
    )


def _case_metric_value(metrics: dict[str, float], name: str) -> float | None:
    value = metrics.get(name)
    return None if value is None else float(value)


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
    "adaptive/low_threshold_fraction",
    "adaptive/low_threshold_count",
    "adaptive/base_WT_pred_foreground_ratio_mean",
    "adaptive/base_WT_pred_foreground_ratio_min",
    "adaptive/base_WT_pred_foreground_ratio_max",
    "adaptive/wt_ratio_threshold",
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


def _case_record(
    case_id: str,
    batch: dict[str, Any],
    batch_index: int,
    target: Tensor,
    threshold: float,
    variants: dict[str, Tensor],
    *,
    region_thresholds: dict[str, float] | None = None,
    adaptive_region_thresholds: tuple[dict[str, float], dict[str, float], float] | None = None,
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
    variant_stability_scores: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case_id,
        "tumor_type": _metadata_raw_value(batch, "Tumor Type", batch_index),
        "tumor_grade": _metadata_raw_value(batch, "Tumor Grade", batch_index),
    }
    for variant_name, logits in variants.items():
        item_metrics = _case_metrics(
            logits[batch_index : batch_index + 1],
            target[batch_index : batch_index + 1],
            threshold=threshold,
        )
        for key in _CASE_RECORD_METRIC_KEYS:
            value = _case_metric_value(item_metrics, key)
            if value is not None:
                record[f"{variant_name}/{key}"] = value
        if region_thresholds is not None:
            calibrated_metrics = brats_region_metrics_from_thresholds(
                logits[batch_index : batch_index + 1],
                target[batch_index : batch_index + 1],
                region_thresholds,
            )
            for key in _CASE_RECORD_METRIC_KEYS:
                value = _case_metric_value(calibrated_metrics, key)
                if value is not None:
                    record[f"{variant_name}_region_calibrated/{key}"] = value
        if adaptive_region_thresholds is not None:
            base_thresholds, low_thresholds, wt_ratio_threshold = adaptive_region_thresholds
            adaptive_metrics = brats_region_metrics_from_adaptive_thresholds(
                logits[batch_index : batch_index + 1],
                target[batch_index : batch_index + 1],
                base_thresholds,
                low_thresholds,
                wt_ratio_threshold=wt_ratio_threshold,
            )
            for key in _CASE_RECORD_METRIC_KEYS:
                value = _case_metric_value(adaptive_metrics, key)
                if value is not None:
                    record[f"{variant_name}_adaptive_region_calibrated/{key}"] = value
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
            stability_scores = None
            if variant_stability_scores is not None:
                stability_scores = variant_stability_scores.get(variant_name)
                if stability_scores is not None:
                    stability_scores = stability_scores[batch_index : batch_index + 1]
            plausibility_metrics = brats_region_metrics_from_plausibility_thresholds(
                logits[batch_index : batch_index + 1],
                target[batch_index : batch_index + 1],
                base_thresholds,
                low_thresholds,
                low_stability_wt_ratio_threshold=low_stability_wt_ratio_threshold,
                low_stability_threshold=low_stability_threshold,
                tc_collapse_wt_ratio_min=tc_collapse_wt_ratio_min,
                tc_collapse_wt_ratio_max=tc_collapse_wt_ratio_max,
                tc_collapse_tc_ratio_threshold=tc_collapse_tc_ratio_threshold,
                stability_scores=stability_scores,
            )
            for key in _CASE_RECORD_METRIC_KEYS:
                value = _case_metric_value(plausibility_metrics, key)
                if value is not None:
                    record[f"{variant_name}_plausibility_region_calibrated/{key}"] = value
    return record


def _oracle_style_selector_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    selected: list[dict[str, float]] = []
    for record in records:
        factual = record.get("factual/brats/mean_dice")
        style = record.get("style_tta/brats/mean_dice")
        if factual is None or style is None:
            continue
        variant = "style_tta" if float(style) >= float(factual) else "factual"
        selected.append(
            {
                "brats/mean_dice": float(record[f"{variant}/brats/mean_dice"]),
                "brats/ET/dice": float(record[f"{variant}/brats/ET/dice"]),
                "brats/TC/dice": float(record[f"{variant}/brats/TC/dice"]),
                "brats/WT/dice": float(record[f"{variant}/brats/WT/dice"]),
            }
        )
    if not selected:
        return {}
    return _prefix_metrics(_average_metric_dicts(selected), "oracle_style_selector")


def _parse_style_modalities(spec: str) -> tuple[int, ...] | None:
    text = str(spec).strip()
    if not text or text.lower() == "all":
        return None
    name_to_index = {name.lower(): index for index, name in enumerate(UTSW_MODALITIES)}
    indices: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part.isdigit():
            index = int(part)
        else:
            if part not in name_to_index:
                valid = ", ".join((*UTSW_MODALITIES, "0", "1", "2", "3", "all"))
                raise ValueError(f"Unknown style TTA modality {part!r}. Valid values: {valid}")
            index = name_to_index[part]
        if index < 0 or index >= len(UTSW_MODALITIES):
            raise ValueError(f"Style TTA modality index {index} is out of range for {UTSW_MODALITIES}")
        if index not in indices:
            indices.append(index)
    if not indices:
        return None
    return tuple(indices)


def _scope_style_intervention(factual_image: Tensor, style_image: Tensor, modalities: tuple[int, ...] | None) -> Tensor:
    if modalities is None:
        return style_image
    scoped = factual_image.clone()
    scoped[:, list(modalities)] = style_image[:, list(modalities)]
    return scoped


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
    model: CausalMedNeXt,
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
    """Counterfactual Context Transport consensus and lower-confidence mask logits.

    CCT keeps the disease representation fixed, transports it across proxy
    contexts, and forms a probability consensus. The stability-gated variant is
    a lower-confidence bound that demotes voxels whose probability depends on
    which proxy context was used.
    """

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


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    splits = checkpoint.get("splits") or _load_json(Path(args.checkpoint).with_name("splits.json"))

    data_root = Path(args.data_root or config.get("data_root", "data/brats/PKG - UTSW-Glioma/UTSW-Glioma"))
    batch_size = int(_config_value(args, config, "batch_size", 1))
    eval_case_ids = _case_ids_from_arg(args.case_ids) or splits[args.split]
    eval_dataset = _make_dataset(data_root, eval_case_ids, args, config)
    should_adjust = _should_build_context_bank(args)
    bank_dataset = _make_dataset(data_root, splits[args.context_split], args, config) if should_adjust else None
    base_use_ants_modalities = bool(_config_value(args, config, "use_ants_modalities", False))
    registered_tta_loader = None
    registered_tta_source = None
    if args.registered_modality_tta:
        registered_tta_source = "native" if base_use_ants_modalities else "ants"
        registered_dataset = _make_dataset_with_modality_source(
            data_root,
            eval_case_ids,
            args,
            config,
            use_ants_modalities=not base_use_ants_modalities,
        )
        registered_tta_loader = _make_loader(registered_dataset, batch_size=batch_size, num_workers=args.num_workers)
    if eval_dataset.metadata_encoder is None and not args.allow_missing_metadata:
        raise FileNotFoundError("Causal MedNeXt evaluation needs metadata proxies. Pass --allow-missing-metadata for representation-only evaluation.")

    eval_loader = _make_loader(eval_dataset, batch_size=batch_size, num_workers=args.num_workers)
    registered_tta_iter = iter(registered_tta_loader) if registered_tta_loader is not None else None
    proxy_layout = _proxy_layout(checkpoint, eval_dataset)

    device = _resolve_device(args.device)
    model = _build_model(checkpoint, eval_dataset, args, config).to(device)
    model.eval()
    style_modalities = _parse_style_modalities(args.style_tta_modalities)
    region_thresholds = parse_region_thresholds(args.region_thresholds)
    adaptive_base_thresholds = parse_region_thresholds(args.adaptive_region_base_thresholds)
    adaptive_low_thresholds = parse_region_thresholds(args.adaptive_region_low_thresholds)
    adaptive_low_candidates = parse_threshold_candidates(args.adaptive_region_low_threshold_candidates)
    adaptive_wt_ratio_candidates = parse_fraction_candidates(args.adaptive_region_wt_ratio_candidates)
    if adaptive_low_thresholds is not None and adaptive_base_thresholds is None:
        raise ValueError("--adaptive-region-base-thresholds is required with --adaptive-region-low-thresholds.")
    if adaptive_base_thresholds is not None and adaptive_low_thresholds is None and not adaptive_low_candidates:
        raise ValueError(
            "--adaptive-region-low-thresholds or --adaptive-region-low-threshold-candidates is required with --adaptive-region-base-thresholds."
        )
    adaptive_region_thresholds = None
    if adaptive_base_thresholds is not None and adaptive_low_thresholds is not None:
        adaptive_region_thresholds = (
            adaptive_base_thresholds,
            adaptive_low_thresholds,
            float(args.adaptive_region_wt_ratio_threshold),
        )
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
    if (adaptive_low_candidates or adaptive_wt_ratio_candidates) and adaptive_base_thresholds is None:
        raise ValueError("--adaptive-region-base-thresholds is required for adaptive threshold sweeps.")
    if bool(adaptive_low_candidates) != bool(adaptive_wt_ratio_candidates):
        raise ValueError(
            "--adaptive-region-low-threshold-candidates and --adaptive-region-wt-ratio-candidates must be supplied together."
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
    style_tta_calibration_sweep = (
        BratsRegionThresholdSweep(calibration_candidates, objective=args.calibration_objective)
        if calibration_candidates
        else None
    )
    registered_tta_calibration_sweep = (
        BratsRegionThresholdSweep(calibration_candidates, objective=args.calibration_objective)
        if calibration_candidates
        else None
    )
    mirror_axes = parse_mirror_tta_axes(args.mirror_tta_axes)
    adaptive_region_sweep = (
        BratsAdaptiveRegionThresholdSweep(
            adaptive_base_thresholds,
            adaptive_low_candidates,
            adaptive_wt_ratio_candidates,
        )
        if adaptive_base_thresholds is not None and adaptive_low_candidates and adaptive_wt_ratio_candidates
        else None
    )
    registered_tta_adaptive_region_sweep = (
        BratsAdaptiveRegionThresholdSweep(
            adaptive_base_thresholds,
            adaptive_low_candidates,
            adaptive_wt_ratio_candidates,
        )
        if adaptive_base_thresholds is not None and adaptive_low_candidates and adaptive_wt_ratio_candidates
        else None
    )
    context_bank = None
    if bank_dataset is not None:
        bank_loader = _make_loader(bank_dataset, batch_size=batch_size, num_workers=args.num_workers)
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
    factual_adaptive_region_calibrated_metrics: list[dict[str, float]] = []
    factual_plausibility_region_calibrated_metrics: list[dict[str, float]] = []
    structural_batch_metrics: list[dict[str, float]] = []
    adjusted_batch_metrics: list[dict[str, float]] = []
    adjusted_region_calibrated_metrics: list[dict[str, float]] = []
    adjusted_structural_batch_metrics: list[dict[str, float]] = []
    cct_consensus_batch_metrics: list[dict[str, float]] = []
    cct_stability_gated_batch_metrics: list[dict[str, float]] = []
    cct_consensus_region_calibrated_metrics: list[dict[str, float]] = []
    cct_stability_gated_region_calibrated_metrics: list[dict[str, float]] = []
    frontdoor_batch_metrics: list[dict[str, float]] = []
    frontdoor_region_batch_metrics: list[dict[str, float]] = []
    region_causal_batch_metrics: list[dict[str, float]] = []
    nested_causal_batch_metrics: list[dict[str, float]] = []
    nested_causal_region_batch_metrics: list[dict[str, float]] = []
    style_tta_batch_metrics: list[dict[str, float]] = []
    style_tta_region_calibrated_metrics: list[dict[str, float]] = []
    registered_tta_batch_metrics: list[dict[str, float]] = []
    registered_tta_region_calibrated_metrics: list[dict[str, float]] = []
    registered_tta_adaptive_region_calibrated_metrics: list[dict[str, float]] = []
    registered_tta_plausibility_region_calibrated_metrics: list[dict[str, float]] = []
    registered_consistency_metric_items: list[dict[str, float]] = []
    registered_stability_scores: list[float] = []
    registered_stability_gate_flags: list[float] = []
    registered_stability_gate_cases: list[str] = []
    factual_volume_metrics: list[dict[str, float]] = []
    adjusted_volume_metrics: list[dict[str, float]] = []
    cct_consensus_volume_metrics: list[dict[str, float]] = []
    cct_stability_gated_volume_metrics: list[dict[str, float]] = []
    cct_instability_items: list[dict[str, float]] = []
    style_tta_volume_metrics: list[dict[str, float]] = []
    registered_tta_volume_metrics: list[dict[str, float]] = []
    nonenhancing_core_completion_cases: list[str] = []
    case_records: list[dict[str, Any]] = []
    proxy_metric_items: list[dict[str, float]] = []
    veto_metric_items: list[dict[str, float]] = []
    context_shifts: list[float] = []
    nearest_context_distances: list[float] = []

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"mednext-causal-eval:{args.split}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        batch_case_ids = _batch_case_ids(batch, image.shape[0])
        registered_batch_stability: Tensor | None = None
        registered_batch_gate_mask: Tensor | None = None
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
        if args.nested_region_consistency:
            logits = _apply_nested_region_consistency(logits)
        if mirror_axes:
            def _forward_factual(augmented: Tensor) -> Tensor:
                tta_outputs = model(
                    augmented,
                    context_bank=bank_device,
                    max_adjustment_contexts=args.adjustment_contexts,
                    adjustment_context_selection=args.adjustment_context_selection,
                )
                tta_logits = tta_outputs["logits"]
                if not isinstance(tta_logits, Tensor):
                    raise TypeError("CausalMedNeXt mirror TTA output 'logits' must be a tensor.")
                if args.nested_region_consistency:
                    tta_logits = _apply_nested_region_consistency(tta_logits)
                return tta_logits

            logits = mirror_tta_logits(_forward_factual, image, mirror_axes, base_logits=logits)

        factual_losses.append(float(_segmentation_loss(logits, target).detach().cpu()))
        logits_cpu = logits.detach().cpu()
        target_cpu = target.detach().cpu()
        factual_batch_metrics.append(brats_region_metrics(logits_cpu, target_cpu, threshold=args.threshold))
        if region_thresholds is not None:
            factual_region_calibrated_metrics.append(brats_region_metrics_from_thresholds(logits_cpu, target_cpu, region_thresholds))
        if adaptive_region_thresholds is not None:
            base_thresholds, low_thresholds, wt_ratio_threshold = adaptive_region_thresholds
            factual_adaptive_region_calibrated_metrics.append(
                brats_region_metrics_from_adaptive_thresholds(
                    logits_cpu,
                    target_cpu,
                    base_thresholds,
                    low_thresholds,
                    wt_ratio_threshold=wt_ratio_threshold,
                )
            )
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
        if adaptive_region_sweep is not None:
            adaptive_region_sweep.update(logits_cpu, target_cpu)
        if calibration_sweep is not None:
            calibration_sweep.update(logits_cpu, target_cpu)
        frontdoor_logits = outputs.get("frontdoor_logits")
        if isinstance(frontdoor_logits, Tensor):
            frontdoor_batch_metrics.append(
                brats_region_metrics(frontdoor_logits.detach().cpu(), target_cpu, threshold=args.threshold)
            )
        frontdoor_region_logits = outputs.get("frontdoor_region_logits")
        if isinstance(frontdoor_region_logits, Tensor):
            frontdoor_region_batch_metrics.append(
                brats_region_metrics_from_region_logits(
                    frontdoor_region_logits.detach().cpu(),
                    target_cpu,
                    threshold=args.threshold,
                )
            )
        region_causal_logits = outputs.get("region_causal_logits")
        if isinstance(region_causal_logits, Tensor):
            region_causal_batch_metrics.append(
                brats_region_metrics(region_causal_logits.detach().cpu(), target_cpu, threshold=args.threshold)
            )
        nested_causal_logits = outputs.get("nested_causal_subregion_logits")
        if isinstance(nested_causal_logits, Tensor):
            nested_causal_batch_metrics.append(
                brats_region_metrics(nested_causal_logits.detach().cpu(), target_cpu, threshold=args.threshold)
            )
        nested_causal_region_logits = outputs.get("nested_causal_region_logits")
        if isinstance(nested_causal_region_logits, Tensor):
            nested_causal_region_batch_metrics.append(
                brats_region_metrics_from_region_logits(
                    nested_causal_region_logits.detach().cpu(),
                    target_cpu,
                    threshold=args.threshold,
                )
            )
        if args.structural_prior:
            structural_batch_metrics.append(
                brats_structural_region_metrics(
                    logits_cpu,
                    target_cpu,
                    threshold=args.structural_threshold,
                    min_component_size=args.structural_min_component_size,
                    fill_holes=args.structural_fill_holes,
                    keep_largest=args.structural_keep_largest,
                )
            )
        factual_volume_metrics.extend(_volume_metrics(logits, target, args.threshold))
        proxy_metric_items.append(_proxy_losses(outputs, batch, proxy_layout))
        veto_metric_items.append(_et_volume_veto_metric_item(outputs))

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
            if args.nested_region_consistency:
                cct_consensus_logits = _apply_nested_region_consistency(cct_consensus_logits)
                cct_stability_gated_logits = _apply_nested_region_consistency(cct_stability_gated_logits)
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

        if args.style_tta_samples > 0:
            tta_logits = [logits]
            rng_devices = _style_rng_devices(device)
            for sample_index in range(int(args.style_tta_samples)):
                if args.deterministic_style_tta:
                    style_seed = _stable_style_seed(args.seed, sample_index, batch_case_ids)
                    with torch.random.fork_rng(devices=rng_devices):
                        torch.manual_seed(style_seed)
                        style_image = apply_style_intervention(image, args)
                else:
                    style_image = apply_style_intervention(image, args)
                style_image = _scope_style_intervention(image, style_image, style_modalities)
                style_outputs = model(style_image, context_bank=None)
                style_logits = style_outputs["logits"]
                if not isinstance(style_logits, Tensor):
                    raise TypeError("CausalMedNeXt style TTA output 'logits' must be a tensor.")
                tta_logits.append(style_logits)
            style_tta_logits = torch.stack(tta_logits, dim=0).mean(dim=0)
            style_tta_logits = _fuse_style_tta_logits(logits, style_tta_logits, args.style_tta_fusion, batch=batch)
            if args.nonenhancing_core_completion:
                style_tta_logits, completion_mask = _apply_nonenhancing_core_completion(
                    style_tta_logits,
                    batch,
                    threshold=args.threshold,
                    fraction=args.nonenhancing_core_fraction,
                    min_wt_voxels=args.nonenhancing_core_min_wt_voxels,
                    max_tc_voxels=args.nonenhancing_core_max_tc_voxels,
                    metadata_gate=args.nonenhancing_core_metadata_gate,
                )
                nonenhancing_core_completion_cases.extend(
                    case_id for case_id, did_complete in zip(batch_case_ids, completion_mask, strict=True) if did_complete
                )
            if args.nested_region_consistency:
                style_tta_logits = _apply_nested_region_consistency(style_tta_logits)
            style_tta_cpu = style_tta_logits.detach().cpu()
            style_tta_batch_metrics.append(brats_region_metrics(style_tta_cpu, target_cpu, threshold=args.threshold))
            if region_thresholds is not None:
                style_tta_region_calibrated_metrics.append(brats_region_metrics_from_thresholds(style_tta_cpu, target_cpu, region_thresholds))
            if style_tta_calibration_sweep is not None:
                style_tta_calibration_sweep.update(style_tta_cpu, target_cpu)
            style_tta_volume_metrics.extend(_volume_metrics(style_tta_logits, target, args.threshold))
        else:
            style_tta_logits = None

        if registered_tta_iter is not None:
            registered_batch = next(registered_tta_iter)
            registered_case_ids = _batch_case_ids(registered_batch, image.shape[0])
            if registered_case_ids != batch_case_ids:
                raise ValueError(
                    f"Registered modality TTA case mismatch: {registered_case_ids} vs {batch_case_ids}"
                )
            registered_image = registered_batch["image"].to(device)
            registered_outputs = model(registered_image, context_bank=None, max_adjustment_contexts=0)
            registered_logits = registered_outputs["logits"]
            if not isinstance(registered_logits, Tensor):
                raise TypeError("CausalMedNeXt registered modality TTA output 'logits' must be a tensor.")
            if args.nested_region_consistency:
                registered_logits = _apply_nested_region_consistency(registered_logits)
            needs_registered_stability = (
                args.registered_modality_fusion == "stability-gated-registered"
                or plausibility_region_thresholds is not None
            )
            if needs_registered_stability:
                registered_batch_stability, registered_batch_gate_mask = _registered_modality_stability_gate_mask(
                    logits,
                    registered_logits,
                    stability_gate_threshold=args.registered_modality_stability_gate_threshold,
                )
                if args.registered_modality_fusion == "stability-gated-registered":
                    stability_values = registered_batch_stability.detach().cpu().tolist()
                    gate_values = registered_batch_gate_mask.detach().cpu().tolist()
                    registered_stability_scores.extend(float(value) for value in stability_values)
                    registered_stability_gate_flags.extend(1.0 if bool(value) else 0.0 for value in gate_values)
                    registered_stability_gate_cases.extend(
                        case_id for case_id, did_gate in zip(batch_case_ids, gate_values, strict=True) if bool(did_gate)
                    )
            registered_tta_logits = _fuse_registered_modality_logits(
                logits,
                registered_logits,
                args.registered_modality_fusion,
                stability_gate_threshold=args.registered_modality_stability_gate_threshold,
            )
            registered_consistency_metric_items.append(
                registered_modality_consistency_metrics(
                    logits,
                    registered_logits,
                    fused_logits=registered_tta_logits,
                    threshold=float(args.threshold),
                )
            )
            registered_tta_cpu = registered_tta_logits.detach().cpu()
            registered_tta_batch_metrics.append(brats_region_metrics(registered_tta_cpu, target_cpu, threshold=args.threshold))
            if region_thresholds is not None:
                registered_tta_region_calibrated_metrics.append(
                    brats_region_metrics_from_thresholds(registered_tta_cpu, target_cpu, region_thresholds)
                )
            if adaptive_region_thresholds is not None:
                base_thresholds, low_thresholds, wt_ratio_threshold = adaptive_region_thresholds
                registered_tta_adaptive_region_calibrated_metrics.append(
                    brats_region_metrics_from_adaptive_thresholds(
                        registered_tta_cpu,
                        target_cpu,
                        base_thresholds,
                        low_thresholds,
                        wt_ratio_threshold=wt_ratio_threshold,
                    )
                )
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
                registered_tta_plausibility_region_calibrated_metrics.append(
                    brats_region_metrics_from_plausibility_thresholds(
                        registered_tta_cpu,
                        target_cpu,
                        base_thresholds,
                        low_thresholds,
                        low_stability_wt_ratio_threshold=low_stability_wt_ratio_threshold,
                        low_stability_threshold=low_stability_threshold,
                        tc_collapse_wt_ratio_min=tc_collapse_wt_ratio_min,
                        tc_collapse_wt_ratio_max=tc_collapse_wt_ratio_max,
                        tc_collapse_tc_ratio_threshold=tc_collapse_tc_ratio_threshold,
                        stability_scores=registered_batch_stability,
                    )
                )
            if registered_tta_adaptive_region_sweep is not None:
                registered_tta_adaptive_region_sweep.update(registered_tta_cpu, target_cpu)
            if registered_tta_calibration_sweep is not None:
                registered_tta_calibration_sweep.update(registered_tta_cpu, target_cpu)
            registered_tta_volume_metrics.extend(_volume_metrics(registered_tta_logits, target, args.threshold))
        else:
            registered_tta_logits = None

        z_c = outputs["z_c"]
        if isinstance(z_c, Tensor):
            nearest_context_distances.extend(_context_overlap(z_c, bank_device).get("overlap/nearest_context_l2", []))

        adjusted = outputs.get("adjusted_logits")
        if isinstance(adjusted, Tensor):
            if args.nested_region_consistency:
                adjusted = _apply_nested_region_consistency(adjusted)
            if mirror_axes:
                def _forward_adjusted(augmented: Tensor) -> Tensor:
                    tta_outputs = model(
                        augmented,
                        context_bank=bank_device,
                        max_adjustment_contexts=args.adjustment_contexts,
                        adjustment_context_selection=args.adjustment_context_selection,
                    )
                    tta_adjusted = tta_outputs["adjusted_logits"]
                    if not isinstance(tta_adjusted, Tensor):
                        raise TypeError("CausalMedNeXt mirror TTA output 'adjusted_logits' must be a tensor.")
                    if args.nested_region_consistency:
                        tta_adjusted = _apply_nested_region_consistency(tta_adjusted)
                    return tta_adjusted

                adjusted = mirror_tta_logits(_forward_adjusted, image, mirror_axes, base_logits=adjusted)
            adjusted_losses.append(float(_segmentation_loss(adjusted, target).detach().cpu()))
            adjusted_cpu = adjusted.detach().cpu()
            adjusted_batch_metrics.append(brats_region_metrics(adjusted_cpu, target_cpu, threshold=args.threshold))
            if region_thresholds is not None:
                adjusted_region_calibrated_metrics.append(brats_region_metrics_from_thresholds(adjusted_cpu, target_cpu, region_thresholds))
            if adjusted_calibration_sweep is not None:
                adjusted_calibration_sweep.update(adjusted_cpu, target_cpu)
            if args.structural_prior:
                adjusted_structural_batch_metrics.append(
                    brats_structural_region_metrics(
                        adjusted_cpu,
                        target_cpu,
                        threshold=args.structural_threshold,
                        min_component_size=args.structural_min_component_size,
                        fill_holes=args.structural_fill_holes,
                        keep_largest=args.structural_keep_largest,
                    )
                )
            adjusted_volume_metrics.extend(_volume_metrics(adjusted, target, args.threshold))
            context_shifts.append(float((torch.sigmoid(logits) - torch.sigmoid(adjusted)).abs().mean().detach().cpu()))

        if args.include_per_case:
            variants: dict[str, Tensor] = {"factual": logits}
            if isinstance(frontdoor_logits, Tensor):
                variants["frontdoor"] = frontdoor_logits
            if isinstance(region_causal_logits, Tensor):
                variants["region_causal"] = region_causal_logits
            nested_causal_logits = outputs.get("nested_causal_subregion_logits")
            if isinstance(nested_causal_logits, Tensor):
                variants["nested_causal"] = nested_causal_logits
            if isinstance(adjusted, Tensor):
                variants["adjusted"] = adjusted
            if isinstance(cct_consensus_logits, Tensor):
                variants["cct_consensus"] = cct_consensus_logits
            if isinstance(cct_stability_gated_logits, Tensor):
                variants["cct_stability_gated"] = cct_stability_gated_logits
            if isinstance(style_tta_logits, Tensor):
                variants["style_tta"] = style_tta_logits
            if isinstance(registered_tta_logits, Tensor):
                variants["registered_tta"] = registered_tta_logits
            for item_index, case_id in enumerate(batch_case_ids):
                record = _case_record(
                    case_id,
                    batch,
                    item_index,
                    target,
                    args.threshold,
                    variants,
                    region_thresholds=region_thresholds,
                    adaptive_region_thresholds=adaptive_region_thresholds,
                    plausibility_region_thresholds=plausibility_region_thresholds,
                    variant_stability_scores=(
                        {"registered_tta": registered_batch_stability}
                        if registered_batch_stability is not None
                        else None
                    ),
                )
                if registered_batch_stability is not None and registered_batch_gate_mask is not None:
                    record["registered_tta/stability_score"] = float(
                        registered_batch_stability[item_index].detach().cpu()
                    )
                    record["registered_tta/stability_gate"] = bool(
                        registered_batch_gate_mask[item_index].detach().cpu()
                    )
                case_records.append(record)

        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    scm = default_utsw_scm()
    metrics: dict[str, Any] = {
        "method": "Causal MedNeXt on UTSW",
        "causal_question": scm.question.query,
        "causal_estimand": scm.question.estimand,
        "causal_warning": scm.question.warning,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "context_split": args.context_split,
        "threshold": float(args.threshold),
        "mirror_tta_axes": ",".join(str(axis) for axis in mirror_axes),
        "num_cases": float(len(factual_volume_metrics)),
        "context_bank_size": float(0 if context_bank is None else context_bank.shape[0]),
        "adjustment_context_selection": str(args.adjustment_context_selection),
        "seed": int(args.seed),
        "style_tta_samples": int(args.style_tta_samples),
        "deterministic_style_tta": bool(args.deterministic_style_tta),
        "style_tta_fusion": str(args.style_tta_fusion),
        "style_tta_modalities": "all" if style_modalities is None else ",".join(UTSW_MODALITIES[index] for index in style_modalities),
        "registered_modality_tta": bool(args.registered_modality_tta),
        "registered_modality_source": str(registered_tta_source or "none"),
        "registered_modality_fusion": str(args.registered_modality_fusion),
        "registered_modality_stability_gate_threshold": float(args.registered_modality_stability_gate_threshold),
        "cct_contexts": int(args.cct_contexts),
        "cct_selection": str(args.cct_selection),
        "cct_instability_scale": float(args.cct_instability_scale),
        "cct_instability_threshold": float(args.cct_instability_threshold),
        "adaptive_region_base_thresholds": str(args.adaptive_region_base_thresholds or ""),
        "adaptive_region_low_thresholds": str(args.adaptive_region_low_thresholds or ""),
        "adaptive_region_wt_ratio_threshold": float(args.adaptive_region_wt_ratio_threshold),
        "adaptive_region_low_threshold_candidates": str(args.adaptive_region_low_threshold_candidates or ""),
        "adaptive_region_wt_ratio_candidates": str(args.adaptive_region_wt_ratio_candidates or ""),
        "plausibility_region_base_thresholds": str(args.plausibility_region_base_thresholds or ""),
        "plausibility_region_low_thresholds": str(args.plausibility_region_low_thresholds or ""),
        "plausibility_low_stability_wt_ratio_threshold": float(args.plausibility_low_stability_wt_ratio_threshold),
        "plausibility_low_stability_threshold": float(args.plausibility_low_stability_threshold),
        "plausibility_tc_collapse_wt_ratio_min": float(args.plausibility_tc_collapse_wt_ratio_min),
        "plausibility_tc_collapse_wt_ratio_max": float(args.plausibility_tc_collapse_wt_ratio_max),
        "plausibility_tc_collapse_tc_ratio_threshold": float(args.plausibility_tc_collapse_tc_ratio_threshold),
        "nonenhancing_core_completion": bool(args.nonenhancing_core_completion),
        "nonenhancing_core_metadata_gate": bool(args.nonenhancing_core_metadata_gate),
        "nested_region_consistency": bool(args.nested_region_consistency),
        "factual/loss": _mean(factual_losses),
    }
    metrics.update(_average_metric_dicts(factual_batch_metrics))
    if factual_region_calibrated_metrics:
        metrics.update(prefix_metrics(_average_metric_dicts(factual_region_calibrated_metrics), "region_calibrated"))
    if factual_adaptive_region_calibrated_metrics:
        metrics.update(
            prefix_metrics(
                _average_metric_dicts(factual_adaptive_region_calibrated_metrics),
                "adaptive_region_calibrated",
            )
        )
    if factual_plausibility_region_calibrated_metrics:
        metrics.update(
            prefix_metrics(
                _average_metric_dicts(factual_plausibility_region_calibrated_metrics),
                "plausibility_region_calibrated",
            )
        )
    if adaptive_region_sweep is not None:
        metrics.update(prefix_metrics(adaptive_region_sweep.summary(), "adaptive_region_sweep"))
    if calibration_sweep is not None:
        metrics.update(prefix_metrics(calibration_sweep.summary(), "sweep_region_calibrated"))
    if frontdoor_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(frontdoor_batch_metrics), "frontdoor"))
    if frontdoor_region_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(frontdoor_region_batch_metrics), "frontdoor_region"))
    if region_causal_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(region_causal_batch_metrics), "region_causal"))
    if nested_causal_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(nested_causal_batch_metrics), "nested_causal"))
    if nested_causal_region_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(nested_causal_region_batch_metrics), "nested_causal_region"))
    if structural_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(structural_batch_metrics), "structural"))
    metrics.update({f"volume/{key}": value for key, value in _average_metric_dicts(factual_volume_metrics).items()})
    metrics.update(_average_metric_dicts(proxy_metric_items))
    metrics.update(_average_metric_dicts(veto_metric_items))
    if adjusted_losses:
        metrics["adjusted/loss"] = _mean(adjusted_losses)
        metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_batch_metrics), "adjusted"))
        if adjusted_region_calibrated_metrics:
            metrics.update(prefix_metrics(_average_metric_dicts(adjusted_region_calibrated_metrics), "adjusted_region_calibrated"))
        if adjusted_calibration_sweep is not None:
            metrics.update(prefix_metrics(adjusted_calibration_sweep.summary(), "adjusted_sweep_region_calibrated"))
        if adjusted_structural_batch_metrics:
            metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_structural_batch_metrics), "adjusted_structural"))
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
    if style_tta_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(style_tta_batch_metrics), "style_tta"))
        if style_tta_region_calibrated_metrics:
            metrics.update(prefix_metrics(_average_metric_dicts(style_tta_region_calibrated_metrics), "style_tta_region_calibrated"))
        if style_tta_calibration_sweep is not None:
            metrics.update(prefix_metrics(style_tta_calibration_sweep.summary(), "style_tta_sweep_region_calibrated"))
        metrics.update({f"style_tta/volume/{key}": value for key, value in _average_metric_dicts(style_tta_volume_metrics).items()})
        metrics["intervention/style_tta_minus_factual_mean_dice"] = float(metrics.get("style_tta/brats/mean_dice", float("nan"))) - float(metrics.get("brats/mean_dice", float("nan")))
        if args.nonenhancing_core_completion:
            metrics["intervention/nonenhancing_core_completion_count"] = float(len(nonenhancing_core_completion_cases))
            metrics["intervention/nonenhancing_core_completion_cases"] = ",".join(nonenhancing_core_completion_cases)
    if registered_tta_batch_metrics:
        metrics.update(_average_metric_dicts(registered_consistency_metric_items))
        metrics.update(_prefix_metrics(_average_metric_dicts(registered_tta_batch_metrics), "registered_tta"))
        if registered_tta_region_calibrated_metrics:
            metrics.update(
                prefix_metrics(_average_metric_dicts(registered_tta_region_calibrated_metrics), "registered_tta_region_calibrated")
            )
        if registered_tta_adaptive_region_calibrated_metrics:
            metrics.update(
                prefix_metrics(
                    _average_metric_dicts(registered_tta_adaptive_region_calibrated_metrics),
                    "registered_tta_adaptive_region_calibrated",
                )
            )
        if registered_tta_plausibility_region_calibrated_metrics:
            metrics.update(
                prefix_metrics(
                    _average_metric_dicts(registered_tta_plausibility_region_calibrated_metrics),
                    "registered_tta_plausibility_region_calibrated",
                )
            )
        if registered_tta_adaptive_region_sweep is not None:
            metrics.update(
                prefix_metrics(
                    registered_tta_adaptive_region_sweep.summary(),
                    "registered_tta_adaptive_region_sweep",
                )
            )
        if registered_tta_calibration_sweep is not None:
            metrics.update(prefix_metrics(registered_tta_calibration_sweep.summary(), "registered_tta_sweep_region_calibrated"))
        metrics.update({f"registered_tta/volume/{key}": value for key, value in _average_metric_dicts(registered_tta_volume_metrics).items()})
        metrics["intervention/registered_tta_minus_factual_mean_dice"] = float(metrics.get("registered_tta/brats/mean_dice", float("nan"))) - float(metrics.get("brats/mean_dice", float("nan")))
        if registered_stability_scores:
            stability_tensor = torch.tensor(registered_stability_scores, dtype=torch.float32)
            gate_tensor = torch.tensor(registered_stability_gate_flags, dtype=torch.float32)
            metrics["registered_tta/stability_gate_threshold"] = float(args.registered_modality_stability_gate_threshold)
            metrics["registered_tta/stability_score_mean"] = float(stability_tensor.mean())
            metrics["registered_tta/stability_score_min"] = float(stability_tensor.min())
            metrics["registered_tta/stability_score_max"] = float(stability_tensor.max())
            metrics["registered_tta/stability_gate_fraction"] = float(gate_tensor.mean())
            metrics["registered_tta/stability_gate_count"] = float(gate_tensor.sum())
            metrics["registered_tta/stability_gate_cases"] = ",".join(registered_stability_gate_cases)
    if case_records:
        metrics["per_case"] = case_records
        metrics.update(_oracle_style_selector_metrics(case_records))
    if nearest_context_distances:
        distances = torch.tensor(nearest_context_distances, dtype=torch.float32)
        metrics["overlap/nearest_context_l2_mean"] = float(distances.mean())
        metrics["overlap/nearest_context_l2_max"] = float(distances.max())
        metrics["overlap/nearest_context_l2_p90"] = float(torch.quantile(distances, 0.9))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate causal MedNeXt with factual, adjusted, proxy, and overlap metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--context-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--case-ids", help="Comma-separated case ids to evaluate instead of the whole split.")
    parser.add_argument("--data-root")
    parser.add_argument("--metadata-path")
    parser.add_argument("--output-json")
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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--region-thresholds", help="Fixed WT/TC/ET thresholds, e.g. 'WT=0.4,TC=0.5,ET=0.6'.")
    parser.add_argument("--calibration-thresholds", help="Comma-separated WT/TC/ET threshold grid for validation-time sweep.")
    parser.add_argument(
        "--calibration-objective",
        choices=CALIBRATION_OBJECTIVES,
        default="mean",
        help="Objective used when choosing thresholds from --calibration-thresholds.",
    )
    parser.add_argument("--mirror-tta-axes", help="Optional spatial mirror TTA axes: d,h,w or z,y,x.")
    parser.add_argument("--structural-prior", action="store_true")
    parser.add_argument("--structural-threshold", type=float, default=0.1)
    parser.add_argument("--structural-min-component-size", type=int, default=16)
    parser.add_argument("--structural-fill-holes", action="store_true")
    parser.add_argument("--structural-keep-largest", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument(
        "--adjustment-context-selection",
        choices=["uniform", "nearest", "farthest", "diverse-nearest"],
        default="uniform",
        help="How the SCM adjusted logits select proxy contexts for each target case.",
    )
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
        help="How CCT selects proxy contexts for transport.",
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-margin", type=int)
    parser.add_argument("--prefer-manual-seg", action="store_true", default=None)
    parser.add_argument("--use-ants-modalities", action="store_true", default=None)
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--style-tta-samples", type=int, default=0)
    parser.add_argument("--deterministic-style-tta", action="store_true")
    parser.add_argument("--style-tta-modalities", default="all")
    parser.add_argument(
        "--style-tta-fusion",
        choices=(
            "all",
            "enhancing-only",
            "enhancing-confident",
            "enhancing-union",
            "enhancing-intersection",
            "enhancing-demote-core",
            "enhancing-empty-consensus-demote-core",
            "enhancing-component-consensus-demote-core",
            "phenotype-enhancing-demote-core",
        ),
        default="all",
    )
    parser.add_argument("--style-scale-range", default="0.85,1.15")
    parser.add_argument("--style-shift-range", default="-0.10,0.10")
    parser.add_argument("--style-gamma-range", default="0.85,1.20")
    parser.add_argument("--style-bias-strength", type=float, default=0.15)
    parser.add_argument("--style-bias-grid-size", type=int, default=4)
    parser.add_argument("--style-noise-std", type=float, default=0.02)
    parser.add_argument("--style-modality-dropout-prob", type=float, default=0.0)
    parser.add_argument("--style-randconv-layers", type=int, default=0)
    parser.add_argument("--style-randconv-kernel-size", type=int, default=3)
    parser.add_argument("--style-randconv-strength", type=float, default=0.0)
    parser.add_argument("--registered-modality-tta", action="store_true")
    parser.add_argument(
        "--registered-modality-fusion",
        choices=("mean-logits", "mean-probs", "max-probs", "registered-only", "stability-gated-registered"),
        default="mean-probs",
    )
    parser.add_argument("--registered-modality-stability-gate-threshold", type=float, default=0.9)
    parser.add_argument("--adaptive-region-base-thresholds", help="Base WT/TC/ET thresholds for adaptive low-confidence calibration.")
    parser.add_argument("--adaptive-region-low-thresholds", help="Lower WT/TC/ET thresholds used when base WT prediction ratio is small.")
    parser.add_argument("--adaptive-region-wt-ratio-threshold", type=float, default=0.0)
    parser.add_argument("--adaptive-region-low-threshold-candidates", help="Comma-separated grid for sweeping adaptive low WT/TC/ET thresholds.")
    parser.add_argument("--adaptive-region-wt-ratio-candidates", help="Comma-separated grid for sweeping adaptive WT predicted-foreground ratio triggers.")
    parser.add_argument("--plausibility-region-base-thresholds", help="Base WT/TC/ET thresholds for plausibility-gated calibration.")
    parser.add_argument("--plausibility-region-low-thresholds", help="Lower WT/TC/ET thresholds for plausibility-gated calibration.")
    parser.add_argument("--plausibility-low-stability-wt-ratio-threshold", type=float, default=0.0)
    parser.add_argument("--plausibility-low-stability-threshold", type=float, default=0.0)
    parser.add_argument("--plausibility-tc-collapse-wt-ratio-min", type=float, default=0.0)
    parser.add_argument("--plausibility-tc-collapse-wt-ratio-max", type=float, default=0.0)
    parser.add_argument("--plausibility-tc-collapse-tc-ratio-threshold", type=float, default=0.0)
    parser.add_argument("--nonenhancing-core-completion", action="store_true")
    parser.add_argument("--nonenhancing-core-metadata-gate", action="store_true")
    parser.add_argument("--nonenhancing-core-fraction", type=float, default=0.20)
    parser.add_argument("--nonenhancing-core-min-wt-voxels", type=int, default=64)
    parser.add_argument("--nonenhancing-core-max-tc-voxels", type=int, default=0)
    parser.add_argument("--nested-region-consistency", action="store_true")
    parser.add_argument("--include-per-case", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    metrics = evaluate(args)
    output_json = Path(args.output_json) if args.output_json else Path(args.checkpoint).with_name(f"{args.split}_causal_metrics.json")
    _save_json(metrics, output_json)
    print(metrics)
    print({"metrics_json": str(output_json)})


if __name__ == "__main__":
    main()
