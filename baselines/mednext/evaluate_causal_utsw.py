from __future__ import annotations

import argparse
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

from baselines.mednext.causal import CausalMedNeXt, build_causal_mednext
from baselines.mednext.train_causal_utsw import apply_style_intervention
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


def _case_metrics(logits: Tensor, target: Tensor, threshold: float) -> dict[str, float]:
    return brats_region_metrics(
        logits.detach().cpu(),
        target.detach().cpu(),
        threshold=threshold,
    )


def _case_metric_value(metrics: dict[str, float], name: str) -> float | None:
    value = metrics.get(name)
    return None if value is None else float(value)


def _case_record(
    case_id: str,
    batch: dict[str, Any],
    batch_index: int,
    target: Tensor,
    threshold: float,
    variants: dict[str, Tensor],
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
        for key in ("brats/mean_dice", "brats/ET/dice", "brats/TC/dice", "brats/WT/dice"):
            value = _case_metric_value(item_metrics, key)
            if value is not None:
                record[f"{variant_name}/{key}"] = value
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


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    splits = checkpoint.get("splits") or _load_json(Path(args.checkpoint).with_name("splits.json"))

    data_root = Path(args.data_root or config.get("data_root", "data/brats/PKG - UTSW-Glioma/UTSW-Glioma"))
    batch_size = int(_config_value(args, config, "batch_size", 1))
    eval_case_ids = _case_ids_from_arg(args.case_ids) or splits[args.split]
    eval_dataset = _make_dataset(data_root, eval_case_ids, args, config)
    bank_dataset = _make_dataset(data_root, splits[args.context_split], args, config)
    if eval_dataset.metadata_encoder is None and not args.allow_missing_metadata:
        raise FileNotFoundError("Causal MedNeXt evaluation needs metadata proxies. Pass --allow-missing-metadata for representation-only evaluation.")

    eval_loader = _make_loader(eval_dataset, batch_size=batch_size, num_workers=args.num_workers)
    bank_loader = _make_loader(bank_dataset, batch_size=batch_size, num_workers=args.num_workers)
    proxy_layout = _proxy_layout(checkpoint, eval_dataset)

    device = _resolve_device(args.device)
    model = _build_model(checkpoint, eval_dataset, args, config).to(device)
    model.eval()
    style_modalities = _parse_style_modalities(args.style_tta_modalities)
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
    structural_batch_metrics: list[dict[str, float]] = []
    adjusted_batch_metrics: list[dict[str, float]] = []
    adjusted_structural_batch_metrics: list[dict[str, float]] = []
    frontdoor_batch_metrics: list[dict[str, float]] = []
    frontdoor_region_batch_metrics: list[dict[str, float]] = []
    region_causal_batch_metrics: list[dict[str, float]] = []
    nested_causal_batch_metrics: list[dict[str, float]] = []
    nested_causal_region_batch_metrics: list[dict[str, float]] = []
    style_tta_batch_metrics: list[dict[str, float]] = []
    factual_volume_metrics: list[dict[str, float]] = []
    adjusted_volume_metrics: list[dict[str, float]] = []
    style_tta_volume_metrics: list[dict[str, float]] = []
    nonenhancing_core_completion_cases: list[str] = []
    case_records: list[dict[str, Any]] = []
    proxy_metric_items: list[dict[str, float]] = []
    context_shifts: list[float] = []
    nearest_context_distances: list[float] = []

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"mednext-causal-eval:{args.split}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        outputs = model(image, context_bank=bank_device, max_adjustment_contexts=args.adjustment_contexts)
        logits = outputs["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("CausalMedNeXt output 'logits' must be a tensor.")
        if args.nested_region_consistency:
            logits = _apply_nested_region_consistency(logits)

        factual_losses.append(float(_segmentation_loss(logits, target).detach().cpu()))
        logits_cpu = logits.detach().cpu()
        target_cpu = target.detach().cpu()
        factual_batch_metrics.append(brats_region_metrics(logits_cpu, target_cpu, threshold=args.threshold))
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

        if args.style_tta_samples > 0:
            tta_logits = [logits]
            for _ in range(int(args.style_tta_samples)):
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
                case_ids = _batch_case_ids(batch, logits.shape[0])
                nonenhancing_core_completion_cases.extend(
                    case_id for case_id, did_complete in zip(case_ids, completion_mask, strict=True) if did_complete
                )
            if args.nested_region_consistency:
                style_tta_logits = _apply_nested_region_consistency(style_tta_logits)
            style_tta_batch_metrics.append(brats_region_metrics(style_tta_logits.detach().cpu(), target_cpu, threshold=args.threshold))
            style_tta_volume_metrics.extend(_volume_metrics(style_tta_logits, target, args.threshold))
        else:
            style_tta_logits = None

        z_c = outputs["z_c"]
        if isinstance(z_c, Tensor):
            nearest_context_distances.extend(_context_overlap(z_c, bank_device).get("overlap/nearest_context_l2", []))

        adjusted = outputs.get("adjusted_logits")
        if isinstance(adjusted, Tensor):
            if args.nested_region_consistency:
                adjusted = _apply_nested_region_consistency(adjusted)
            adjusted_losses.append(float(_segmentation_loss(adjusted, target).detach().cpu()))
            adjusted_cpu = adjusted.detach().cpu()
            adjusted_batch_metrics.append(brats_region_metrics(adjusted_cpu, target_cpu, threshold=args.threshold))
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
            if isinstance(style_tta_logits, Tensor):
                variants["style_tta"] = style_tta_logits
            for item_index, case_id in enumerate(_batch_case_ids(batch, logits.shape[0])):
                case_records.append(
                    _case_record(
                        case_id,
                        batch,
                        item_index,
                        target,
                        args.threshold,
                        variants,
                    )
                )

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
        "num_cases": float(len(factual_volume_metrics)),
        "context_bank_size": float(0 if context_bank is None else context_bank.shape[0]),
        "seed": int(args.seed),
        "style_tta_samples": int(args.style_tta_samples),
        "style_tta_fusion": str(args.style_tta_fusion),
        "style_tta_modalities": "all" if style_modalities is None else ",".join(UTSW_MODALITIES[index] for index in style_modalities),
        "nonenhancing_core_completion": bool(args.nonenhancing_core_completion),
        "nonenhancing_core_metadata_gate": bool(args.nonenhancing_core_metadata_gate),
        "nested_region_consistency": bool(args.nested_region_consistency),
        "factual/loss": _mean(factual_losses),
    }
    metrics.update(_average_metric_dicts(factual_batch_metrics))
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
    if adjusted_losses:
        metrics["adjusted/loss"] = _mean(adjusted_losses)
        metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_batch_metrics), "adjusted"))
        if adjusted_structural_batch_metrics:
            metrics.update(_prefix_metrics(_average_metric_dicts(adjusted_structural_batch_metrics), "adjusted_structural"))
        metrics.update({f"adjusted/volume/{key}": value for key, value in _average_metric_dicts(adjusted_volume_metrics).items()})
        metrics["intervention/context_adjustment_mean_abs_prob_shift"] = _mean(context_shifts)
        metrics["intervention/adjusted_minus_factual_mean_dice"] = float(metrics.get("adjusted/brats/mean_dice", float("nan"))) - float(metrics.get("brats/mean_dice", float("nan")))
    if style_tta_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(style_tta_batch_metrics), "style_tta"))
        metrics.update({f"style_tta/volume/{key}": value for key, value in _average_metric_dicts(style_tta_volume_metrics).items()})
        metrics["intervention/style_tta_minus_factual_mean_dice"] = float(metrics.get("style_tta/brats/mean_dice", float("nan"))) - float(metrics.get("brats/mean_dice", float("nan")))
        if args.nonenhancing_core_completion:
            metrics["intervention/nonenhancing_core_completion_count"] = float(len(nonenhancing_core_completion_cases))
            metrics["intervention/nonenhancing_core_completion_cases"] = ",".join(nonenhancing_core_completion_cases)
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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-margin", type=int)
    parser.add_argument("--prefer-manual-seg", action="store_true", default=None)
    parser.add_argument("--use-ants-modalities", action="store_true", default=None)
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--style-tta-samples", type=int, default=0)
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
