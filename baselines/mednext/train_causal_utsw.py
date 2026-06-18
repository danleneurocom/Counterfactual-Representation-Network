from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from baselines.mednext.calibration import (
    CALIBRATION_OBJECTIVES,
    BratsRegionThresholdSweep,
    brats_region_probabilities,
    brats_region_targets,
    parse_threshold_candidates,
    prefix_metrics,
)
from baselines.mednext.causal import CausalMedNeXt, build_causal_mednext
from baselines.mednext.common import (
    _parse_channel_weights,
    checkpoint_monitor,
    load_model_for_eval,
    main_logits,
    registered_modality_consistency_metrics,
    segmentation_terms as _segmentation_terms,
)
from baselines.mednext.dataset_cache import maybe_disk_cache_dataset
from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_causal_utsw import (
    _add_context_swap_outputs,
    _build_optimizer,
    _causal_loss_terms as _shared_causal_loss_terms,
    _float_terms,
    _load_json,
    _metadata_dims,
    _metadata_layout,
    _prefix_metrics,
    _boundary_mediator_loss,
    _prototype_mediator_loss,
    _probability_consistency_loss,
    _require_metadata,
    _set_backbone_trainable,
    _spatial_disease_attention_loss,
    _spatial_region_head_loss,
    _subsample_context_bank,
    _weighted_total as _shared_weighted_total,
)
from baselines.segformer3d.train_utsw import _average_metric_dicts, _case_ids, _make_splits, _resolve_device, _save_json
from crn.metrics import brats_region_metrics


def _et_volume_veto_metric_item(outputs: dict[str, Tensor | tuple[Tensor, ...]]) -> dict[str, float]:
    bias = outputs.get("et_volume_veto_bias")
    if not isinstance(bias, Tensor):
        return {}
    item = {
        "et_volume_veto/bias_mean": float(bias.detach().mean().cpu()),
        "et_volume_veto/bias_max": float(bias.detach().max().cpu()),
        "et_volume_veto/active_fraction": float((bias.detach() > 0).float().mean().cpu()),
    }
    for key, name in (
        ("et_volume_veto_predicted_fraction", "predicted_fraction"),
        ("et_volume_veto_allowed_fraction", "allowed_fraction"),
        ("et_volume_proxy_fraction", "proxy_fraction"),
    ):
        value = outputs.get(key)
        if isinstance(value, Tensor):
            item[f"et_volume_veto/{name}_mean"] = float(value.detach().mean().cpu())
    return item


def _et_volume_veto_scale_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    target = max(0.0, float(getattr(args, "et_volume_veto_scale", 0.0)))
    if target <= 0.0:
        return 0.0
    warmup_epochs = max(0, int(getattr(args, "et_volume_veto_warmup_epochs", 0)))
    if int(epoch) <= warmup_epochs:
        return 0.0
    ramp_epochs = max(0, int(getattr(args, "et_volume_veto_ramp_epochs", 0)))
    if ramp_epochs <= 0:
        return target
    progress = min(max(int(epoch) - warmup_epochs, 0), ramp_epochs) / float(ramp_epochs)
    return target * progress


def _set_et_volume_veto_scale_for_epoch(model: CausalMedNeXt, args: argparse.Namespace, epoch: int) -> float:
    scale = _et_volume_veto_scale_for_epoch(args, epoch)
    model.et_volume_veto_scale = scale
    return scale


class RegisteredModalityPairDataset(Dataset):
    """Adds the opposite native/ANTs image for the same UTSW case."""

    def __init__(self, primary: Dataset, registered: Dataset) -> None:
        if len(primary) != len(registered):
            raise ValueError(f"Registered modality pair length mismatch: {len(primary)} vs {len(registered)}")
        self.primary = primary
        self.registered = registered

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)

    def __len__(self) -> int:
        return len(self.primary)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.primary[index])
        registered_item = self.registered[index]
        primary_case = item.get("case_id")
        registered_case = registered_item.get("case_id") if isinstance(registered_item, dict) else None
        if primary_case != registered_case:
            raise ValueError(f"Registered modality pair case mismatch at {index}: {primary_case!r} vs {registered_case!r}")
        item["registered_image"] = registered_item["image"]
        item["registered_case_id"] = registered_case
        return item


def _small_lesion_emphasis_weights(
    target: Tensor,
    *,
    emphasis: float,
    reference_fractions: str,
    region_weights: str,
    max_weight: float,
    reference_name: str,
    region_weights_name: str,
) -> Tensor:
    dtype = target.dtype if target.is_floating_point() else torch.float32
    if float(emphasis) <= 0.0:
        return torch.ones((), device=target.device, dtype=dtype)
    if target.ndim == 4:
        target = target.unsqueeze(0)
    target = target.to(dtype=dtype)
    references = _parse_channel_weights(reference_fractions, 3, reference_name)
    weights = _parse_channel_weights(region_weights, 3, region_weights_name)
    region_target = brats_region_targets(target).to(device=target.device, dtype=dtype)
    region_volume = region_target.mean(dim=tuple(range(2, region_target.ndim)))
    reference_tensor = torch.as_tensor(references, device=target.device, dtype=dtype).clamp_min(1e-6)
    region_weight = torch.as_tensor(weights, device=target.device, dtype=dtype).clamp_min(0.0)
    rarity = ((reference_tensor.view(1, 3) - region_volume).clamp_min(0.0) / reference_tensor.view(1, 3)).clamp(0.0, 1.0)
    rarity_score = (rarity * region_weight.view(1, 3)).sum(dim=1).div(region_weight.sum().clamp_min(1e-6))
    non_empty = (region_volume[:, 0] > 0.0).to(dtype=dtype)
    weight = 1.0 + float(emphasis) * rarity_score * non_empty
    return weight.clamp(1.0, max(1.0, float(max_weight))).detach()


def _make_utsw_dataset(
    root: Path,
    case_ids: list[str],
    args: argparse.Namespace,
    cache_name: str,
    *,
    use_ants_modalities: bool,
    cache_suffix: str = "",
) -> Dataset:
    dataset = UTSWGliomaDataset(
        root=root,
        volume_size=args.volume_size,
        case_ids=case_ids,
        crop_margin=args.crop_margin,
        prefer_manual_seg=args.prefer_manual_seg,
        use_ants_modalities=use_ants_modalities,
        metadata_path=args.metadata_path,
        include_metadata=True,
    )
    return maybe_disk_cache_dataset(
        dataset,
        args.disk_cache_dir,
        namespace=f"utsw_causal_{cache_name}{cache_suffix}",
        signature={
            "dataset": "utsw_causal",
            "split": cache_name,
            "root": str(root),
            "volume_size": args.volume_size,
            "crop_margin": args.crop_margin,
            "prefer_manual_seg": args.prefer_manual_seg,
            "use_ants_modalities": use_ants_modalities,
            "metadata_path": args.metadata_path,
            "include_metadata": True,
        },
    )


def _uses_registered_modality_training(args: argparse.Namespace) -> bool:
    return any(
        float(getattr(args, name, 0.0)) > 0.0
        for name in (
            "lambda_registered_modality_seg",
            "lambda_registered_modality_consistency",
            "lambda_registered_modality_region_consistency",
            "lambda_registered_modality_wt_consistency",
            "lambda_registered_modality_fusion_seg",
            "lambda_registered_modality_view_advantage_distillation",
            "lambda_registered_modality_disease_invariance",
        )
    )


def _uses_registered_modality_validation(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "checkpoint_registered_modality_tta", False))


def _hard_case_sampler_weights(dataset: Dataset, args: argparse.Namespace) -> Tensor | None:
    emphasis = float(getattr(args, "hard_case_sampler_emphasis", 0.0) or 0.0)
    if emphasis <= 0.0:
        return None
    source = dataset.primary if isinstance(dataset, RegisteredModalityPairDataset) else dataset
    weights: list[float] = []
    for index in range(len(source)):
        item = source[index]
        target = item["mask"]
        if not isinstance(target, Tensor):
            raise TypeError(f"Expected tensor mask for hard-case sampler at index {index}, got {type(target).__name__}")
        weight = _small_lesion_emphasis_weights(
            target,
            emphasis=emphasis,
            reference_fractions=getattr(args, "hard_case_sampler_reference_fractions", "0.015,0.006,0.003"),
            region_weights=getattr(args, "hard_case_sampler_region_weights", "0.5,1.0,1.5"),
            max_weight=float(getattr(args, "hard_case_sampler_max_weight", 3.0) or 3.0),
            reference_name="--hard-case-sampler-reference-fractions",
            region_weights_name="--hard-case-sampler-region-weights",
        )
        weights.append(float(weight.mean().cpu()))
    if not weights:
        return None
    return torch.as_tensor(weights, dtype=torch.double).clamp_min(1e-6)


def _hard_case_sampler(dataset: Dataset, args: argparse.Namespace) -> WeightedRandomSampler | None:
    weights = _hard_case_sampler_weights(dataset, args)
    if weights is None:
        return None
    multiplier = max(float(getattr(args, "hard_case_sampler_epoch_multiplier", 1.0) or 1.0), 1e-6)
    num_samples = max(1, int(round(len(weights) * multiplier)))
    generator = torch.Generator()
    generator.manual_seed(int(getattr(args, "seed", 7)))
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True, generator=generator)


def _make_loader(
    root: Path,
    case_ids: list[str],
    args: argparse.Namespace,
    shuffle: bool,
    cache_name: str,
    *,
    registered_modality_pair: bool = False,
) -> DataLoader:
    dataset = _make_utsw_dataset(
        root,
        case_ids,
        args,
        cache_name,
        use_ants_modalities=args.use_ants_modalities,
    )
    if registered_modality_pair:
        registered_source = "native" if args.use_ants_modalities else "ants"
        registered_dataset = _make_utsw_dataset(
            root,
            case_ids,
            args,
            cache_name,
            use_ants_modalities=not args.use_ants_modalities,
            cache_suffix=f"_registered_{registered_source}",
        )
        dataset = RegisteredModalityPairDataset(dataset, registered_dataset)
    sampler = _hard_case_sampler(dataset, args) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def _load_or_make_splits(args: argparse.Namespace, baseline_checkpoint: dict[str, Any], data_root: Path) -> dict[str, list[str]]:
    if args.splits_json:
        return _load_json(Path(args.splits_json))
    if args.limit_cases is not None:
        return _make_splits(_case_ids(data_root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)
    if "splits" in baseline_checkpoint:
        return baseline_checkpoint["splits"]
    return _make_splits(_case_ids(data_root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)


def _build_model_from_dataset(args: argparse.Namespace, dataset: UTSWGliomaDataset) -> CausalMedNeXt:
    treatment_proxy_dim = 2
    if dataset.metadata_encoder is not None and hasattr(dataset.metadata_encoder, "treatment_dim"):
        treatment_proxy_dim = int(dataset.metadata_encoder.treatment_dim)
    return build_causal_mednext(
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        latent_dim=args.latent_dim,
        num_classes=3,
        base_channels=args.base_channels,
        modulation_scale=args.modulation_scale,
        causal_residual_scale=args.causal_residual_scale,
        contrastive_dim=args.contrastive_dim,
        spatial_refiner_scale=args.spatial_refiner_scale,
        region_fusion_scale=args.region_fusion_scale,
        prototype_dim=args.prototype_dim,
        prototype_fusion_scale=args.prototype_fusion_scale,
        prototype_temperature=args.prototype_temperature,
        category_confounder_scale=args.category_confounder_scale,
        category_confounder_temperature=args.category_confounder_temperature,
        modality_prior_scale=args.modality_prior_scale,
        logit_calibration_scale=args.logit_calibration_scale,
        cascade_refiner_scale=args.cascade_refiner_scale,
        frontdoor_mediator_scale=args.frontdoor_mediator_scale,
        frontdoor_residual_scale=args.frontdoor_residual_scale,
        use_causal_mediator_router=args.use_causal_mediator_router,
        use_nested_causal_intervention=args.use_nested_causal_intervention,
        nested_causal_gate_scale=args.nested_causal_gate_scale,
        region_causal_bottleneck_scale=args.region_causal_bottleneck_scale,
        region_causal_background_leak=args.region_causal_background_leak,
        region_causal_base=args.region_causal_base,
        region_causal_mask_source=args.region_causal_mask_source,
        region_volume_scale=args.region_volume_scale,
        et_volume_veto_scale=args.et_volume_veto_scale,
        et_volume_veto_multiplier=args.et_volume_veto_multiplier,
        et_volume_veto_min_fraction=args.et_volume_veto_min_fraction,
        et_volume_veto_max_bias=args.et_volume_veto_max_bias,
        treatment_proxy_dim=treatment_proxy_dim,
        **_metadata_dims(dataset),
    )


def _load_baseline_backbone(model: CausalMedNeXt, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_baseline_state_dict(state_dict, strict_backbone=True)
    return checkpoint


def _load_causal_init_checkpoint(model: CausalMedNeXt, checkpoint_path: str | Path | None) -> dict[str, Any]:
    if not checkpoint_path:
        return {}
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    report = model.load_compatible_state_dict(state_dict)
    report["checkpoint"] = str(path)
    report["epoch"] = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    report["loaded_keys"] = len(state_dict) - len(report["unexpected_keys"]) - len(report["skipped_shape_keys"])
    return report


def _build_teacher_model(checkpoint_path: str | Path | None, device: torch.device) -> torch.nn.Module | None:
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    eval_args = argparse.Namespace(model_id=None, kernel_size=None, base_channels=None)
    teacher = load_model_for_eval(checkpoint, eval_args)
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


@torch.no_grad()
def _teacher_logits(teacher: torch.nn.Module | None, image: Tensor) -> Tensor | None:
    if teacher is None:
        return None
    return main_logits(teacher(image)).detach()


def _causal_loss_terms(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    batch: dict[str, Any],
    args: argparse.Namespace,
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Tensor]:
    """Use the shared causal objective with MedNeXt-local segmentation terms."""

    shared_globals = _shared_causal_loss_terms.__globals__
    original_segmentation_terms = shared_globals.get("_segmentation_terms")
    shared_globals["_segmentation_terms"] = _segmentation_terms
    try:
        return _shared_causal_loss_terms(outputs, batch, args, proxy_layout)
    finally:
        if original_segmentation_terms is None:
            shared_globals.pop("_segmentation_terms", None)
        else:
            shared_globals["_segmentation_terms"] = original_segmentation_terms


def _parse_float_range(spec: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(spec).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as 'low,high', got {spec!r}")
    low, high = float(parts[0]), float(parts[1])
    if high < low:
        raise ValueError(f"{name} high value must be >= low value, got {spec!r}")
    return low, high


def _uniform_like(shape: tuple[int, ...], low: float, high: float, image: Tensor) -> Tensor:
    if high == low:
        return torch.full(shape, low, device=image.device, dtype=image.dtype)
    return torch.empty(shape, device=image.device, dtype=image.dtype).uniform_(low, high)


def _match_channel_moments(source: Tensor, reference: Tensor) -> Tensor:
    dims = tuple(range(2, source.ndim))
    source_mean = source.mean(dim=dims, keepdim=True)
    source_std = source.std(dim=dims, keepdim=True).clamp_min(1e-6)
    reference_mean = reference.mean(dim=dims, keepdim=True)
    reference_std = reference.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return (source - source_mean) / source_std * reference_std + reference_mean


def _randconv3d_style(image: Tensor, layers: int, kernel_size: int, strength: float) -> Tensor:
    if layers <= 0 or strength <= 0.0:
        return image
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    styled = image
    batch, channels = styled.shape[:2]
    groups = batch * channels
    for _ in range(int(layers)):
        flat = styled.reshape(1, groups, *styled.shape[-3:])
        weight = torch.randn(
            (groups, 1, kernel_size, kernel_size, kernel_size),
            device=styled.device,
            dtype=styled.dtype,
        )
        weight = weight / weight.flatten(start_dim=1).norm(dim=1).view(groups, 1, 1, 1, 1).clamp_min(1e-6)
        convolved = F.conv3d(flat, weight, padding=kernel_size // 2, groups=groups)
        styled = torch.tanh(convolved.reshape_as(styled))
        styled = _match_channel_moments(styled, image)
    return (1.0 - float(strength)) * image + float(strength) * styled


def apply_style_intervention(image: Tensor, args: argparse.Namespace) -> Tensor:
    """Approximate do(S=s') acquisition interventions while preserving anatomy."""
    augmented = image.clone()
    batch, channels = augmented.shape[:2]
    bc111 = (batch, channels, 1, 1, 1)

    scale_low, scale_high = _parse_float_range(args.style_scale_range, "--style-scale-range")
    shift_low, shift_high = _parse_float_range(args.style_shift_range, "--style-shift-range")
    gamma_low, gamma_high = _parse_float_range(args.style_gamma_range, "--style-gamma-range")

    scale = _uniform_like(bc111, scale_low, scale_high, image)
    shift = _uniform_like(bc111, shift_low, shift_high, image)
    augmented = augmented * scale + shift

    if gamma_low != 1.0 or gamma_high != 1.0:
        gamma = _uniform_like(bc111, gamma_low, gamma_high, image)
        flat = augmented.flatten(start_dim=2)
        min_value = flat.amin(dim=2).view(bc111)
        max_value = flat.amax(dim=2).view(bc111)
        normalized = (augmented - min_value) / (max_value - min_value).clamp_min(1e-6)
        augmented = normalized.clamp(0.0, 1.0).pow(gamma) * (max_value - min_value) + min_value

    bias_strength = float(args.style_bias_strength)
    if bias_strength > 0.0:
        grid_size = max(2, int(args.style_bias_grid_size))
        field = torch.randn(
            (batch, channels, grid_size, grid_size, grid_size),
            device=image.device,
            dtype=image.dtype,
        )
        field = F.interpolate(field, size=augmented.shape[-3:], mode="trilinear", align_corners=False)
        field = field - field.mean(dim=(2, 3, 4), keepdim=True)
        field = field / field.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        augmented = augmented * torch.exp(bias_strength * field)

    noise_std = float(args.style_noise_std)
    if noise_std > 0.0:
        augmented = augmented + torch.randn_like(augmented) * noise_std

    augmented = _randconv3d_style(
        augmented,
        layers=int(getattr(args, "style_randconv_layers", 0)),
        kernel_size=int(getattr(args, "style_randconv_kernel_size", 3)),
        strength=float(getattr(args, "style_randconv_strength", 0.0)),
    )

    dropout_prob = float(args.style_modality_dropout_prob)
    if dropout_prob > 0.0:
        keep = (torch.rand((batch, channels, 1, 1, 1), device=image.device) > dropout_prob).to(dtype=image.dtype)
        all_dropped = keep.sum(dim=1, keepdim=True) == 0
        keep = torch.where(all_dropped, torch.ones_like(keep), keep)
        augmented = augmented * keep

    return augmented.clamp(-6.0, 6.0)


def _apply_causal_feature_mask(features: tuple[Tensor, ...], args: argparse.Namespace) -> tuple[Tensor, ...]:
    masked: list[Tensor] = []
    probability = float(args.feature_mask_prob)
    block_size = max(1, int(args.feature_mask_block_size))
    for feature in features:
        channel_keep = (torch.rand((feature.shape[0], feature.shape[1], 1, 1, 1), device=feature.device) > probability).to(
            dtype=feature.dtype
        )
        if block_size > 1 and min(feature.shape[-3:]) > 1:
            low_size = tuple(max(1, int((dim + block_size - 1) // block_size)) for dim in feature.shape[-3:])
            spatial_keep = (torch.rand((feature.shape[0], 1, *low_size), device=feature.device) > probability).to(
                dtype=feature.dtype
            )
            spatial_keep = F.interpolate(spatial_keep, size=feature.shape[-3:], mode="nearest")
        else:
            spatial_keep = (torch.rand((feature.shape[0], 1, *feature.shape[-3:]), device=feature.device) > probability).to(
                dtype=feature.dtype
            )
        keep = torch.maximum(channel_keep, spatial_keep)
        masked.append(feature * keep / keep.mean().clamp_min(0.2))
    return tuple(masked)


@dataclass
class LesionPatch:
    image: Tensor
    mask: Tensor


class LesionInterventionBank:
    """Stores observed lesion patches for do(D=1) paste and do(D=0) erase interventions."""

    def __init__(
        self,
        max_patches: int = 32,
        min_voxels: int = 8,
        edge_softening: int = 3,
        min_brain_coverage: float = 0.0,
        placement_attempts: int = 8,
        match_recipient_moments: bool = False,
    ) -> None:
        self.max_patches = max(1, int(max_patches))
        self.min_voxels = max(1, int(min_voxels))
        self.edge_softening = max(1, int(edge_softening))
        self.min_brain_coverage = float(min(max(min_brain_coverage, 0.0), 1.0))
        self.placement_attempts = max(1, int(placement_attempts))
        self.match_recipient_moments = bool(match_recipient_moments)
        if self.edge_softening % 2 == 0:
            self.edge_softening += 1
        self._patches: list[LesionPatch] = []

    def __len__(self) -> int:
        return len(self._patches)

    @staticmethod
    def _crop_to_volume(patch: Tensor, shape: tuple[int, int, int]) -> Tensor:
        slices = []
        for size, max_size in zip(patch.shape[-3:], shape, strict=True):
            if size <= max_size:
                slices.append(slice(0, size))
                continue
            start = int((size - max_size) // 2)
            slices.append(slice(start, start + max_size))
        return patch[(..., *slices)]

    def _alpha(self, mask: Tensor) -> Tensor:
        hard = mask.amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        if self.edge_softening <= 1:
            return hard
        padding = self.edge_softening // 2
        dilated = F.max_pool3d(hard, self.edge_softening, stride=1, padding=padding)
        return F.avg_pool3d(dilated, self.edge_softening, stride=1, padding=padding).clamp(0.0, 1.0)

    def update(self, image: Tensor, target: Tensor) -> None:
        image_cpu = image.detach().cpu()
        target_cpu = target.detach().cpu()
        spatial_shape = tuple(int(value) for value in target_cpu.shape[-3:])
        for index in range(target_cpu.shape[0]):
            lesion = target_cpu[index].amax(dim=0) > 0.5
            if int(lesion.sum().item()) < self.min_voxels:
                continue
            coords = lesion.nonzero(as_tuple=False)
            low = coords.min(dim=0).values
            high = coords.max(dim=0).values + 1
            pad = 2
            low = torch.maximum(low - pad, torch.zeros_like(low))
            high = torch.minimum(high + pad, torch.as_tensor(spatial_shape, dtype=high.dtype))
            slices = tuple(slice(int(start), int(stop)) for start, stop in zip(low, high, strict=True))
            patch_image = image_cpu[index : index + 1, :, *slices].contiguous()
            patch_mask = target_cpu[index : index + 1, :, *slices].contiguous()
            if int(patch_mask.amax(dim=1).sum().item()) < self.min_voxels:
                continue
            self._patches.append(LesionPatch(patch_image, patch_mask))
        if len(self._patches) > self.max_patches:
            self._patches = self._patches[-self.max_patches :]

    def paste(self, image: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor] | None:
        if not self._patches:
            return None
        patch = random.choice(self._patches)
        patch_image = patch.image.to(device=image.device, dtype=image.dtype)
        patch_mask = patch.mask.to(device=target.device, dtype=target.dtype)
        volume_shape = tuple(int(value) for value in image.shape[-3:])
        patch_image = self._crop_to_volume(patch_image, volume_shape)
        patch_mask = self._crop_to_volume(patch_mask, volume_shape)
        patch_shape = tuple(int(value) for value in patch_mask.shape[-3:])
        if int(patch_mask.amax(dim=1).sum().item()) < self.min_voxels:
            return None
        hard_alpha = patch_mask.amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        starts: list[int] | None = None
        for _ in range(self.placement_attempts):
            candidate = [
                random.randint(0, max(0, size - patch_size))
                for size, patch_size in zip(volume_shape, patch_shape, strict=True)
            ]
            if self.min_brain_coverage <= 0.0:
                starts = candidate
                break
            candidate_slices = tuple(
                slice(start, start + patch_size)
                for start, patch_size in zip(candidate, patch_shape, strict=True)
            )
            image_region = image[:, :, *candidate_slices]
            brain_region = (image_region.abs().amax(dim=1, keepdim=True) > 0.02).to(dtype=hard_alpha.dtype)
            hard = hard_alpha.expand(image_region.shape[0], -1, -1, -1, -1)
            coverage = (brain_region * hard).sum() / hard.sum().clamp_min(1.0)
            if float(coverage.detach().cpu()) >= self.min_brain_coverage:
                starts = candidate
                break
        if starts is None:
            return None
        slices = tuple(slice(start, start + patch_size) for start, patch_size in zip(starts, patch_shape, strict=True))
        soft_alpha = self._alpha(patch_mask)
        mixed_image = image.clone()
        mixed_target = target.clone()
        effect_mask = torch.zeros_like(target)
        image_region = mixed_image[:, :, *slices]
        target_region = mixed_target[:, :, *slices]
        source_image = patch_image.expand(image_region.shape[0], -1, -1, -1, -1)
        if self.match_recipient_moments:
            source_image = _match_channel_moments(source_image, image_region)
        source_mask = patch_mask.expand(target_region.shape[0], -1, -1, -1, -1)
        hard = hard_alpha.expand(target_region.shape[0], -1, -1, -1, -1)
        soft = soft_alpha.expand(image_region.shape[0], -1, -1, -1, -1)
        mixed_image[:, :, *slices] = image_region * (1.0 - soft) + source_image * soft
        mixed_target[:, :, *slices] = torch.maximum(target_region * (1.0 - hard), source_mask)
        effect_mask[:, :, *slices] = torch.maximum(effect_mask[:, :, *slices], source_mask)
        return mixed_image, mixed_target.clamp(0.0, 1.0), effect_mask.clamp(0.0, 1.0)

    def erase(self, image: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor] | None:
        lesion = target.amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        if int(lesion.sum().item()) < self.min_voxels:
            return None
        context = 1.0 - lesion
        denom = context.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
        mean = (image * context).sum(dim=(2, 3, 4), keepdim=True) / denom
        variance = ((image - mean).pow(2) * context).sum(dim=(2, 3, 4), keepdim=True) / denom
        fill = mean + 0.03 * variance.sqrt().clamp_min(1e-6) * torch.randn_like(image)
        alpha = self._alpha(target)
        erased_image = image * (1.0 - alpha) + fill * alpha
        erased_target = target * (1.0 - lesion)
        return erased_image.clamp(-6.0, 6.0), erased_target.clamp(0.0, 1.0), target * lesion


def _should_apply_style_intervention(args: argparse.Namespace) -> bool:
    if float(args.style_intervention_prob) <= 0.0:
        return False
    if not any(
        float(value) > 0.0
        for value in (
            args.lambda_style_intervention_seg,
            args.lambda_style_intervention_consistency,
            args.lambda_style_disease_invariance,
            args.lambda_style_context_response,
        )
    ):
        return False
    return random.random() < float(args.style_intervention_prob)


def _should_apply_feature_intervention(args: argparse.Namespace) -> bool:
    if float(args.feature_intervention_prob) <= 0.0 or float(args.feature_mask_prob) <= 0.0:
        return False
    if not any(
        float(value) > 0.0
        for value in (
            args.lambda_feature_intervention_seg,
            args.lambda_feature_intervention_consistency,
        )
    ):
        return False
    return random.random() < float(args.feature_intervention_prob)


def _should_apply_lesion_intervention(args: argparse.Namespace) -> bool:
    if float(args.lesion_intervention_prob) <= 0.0:
        return False
    if not any(
        float(value) > 0.0
        for value in (
            args.lambda_lesion_paste_seg,
            args.lambda_lesion_erase_seg,
            args.lambda_lesion_intervention_effect,
        )
    ):
        return False
    return random.random() < float(args.lesion_intervention_prob)


def _style_intervention_terms(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    style_outputs: dict[str, Tensor | tuple[Tensor, ...]],
    target: Tensor,
    args: argparse.Namespace,
) -> dict[str, Tensor]:
    terms: dict[str, Tensor] = {}
    style_logits = style_outputs.get("logits")
    factual_logits = outputs.get("logits")
    if isinstance(style_logits, Tensor) and float(args.lambda_style_intervention_seg) > 0.0:
        terms.update(_segmentation_terms(style_logits, target, args, "style_intervention_seg"))
    if (
        isinstance(style_logits, Tensor)
        and isinstance(factual_logits, Tensor)
        and float(args.lambda_style_intervention_consistency) > 0.0
    ):
        terms["style_intervention_consistency"] = _probability_consistency_loss(style_logits, factual_logits, args)
    z_d = outputs.get("z_d")
    style_z_d = style_outputs.get("z_d")
    if isinstance(z_d, Tensor) and isinstance(style_z_d, Tensor) and float(args.lambda_style_disease_invariance) > 0.0:
        terms["style_disease_invariance"] = F.mse_loss(
            F.normalize(style_z_d, dim=1),
            F.normalize(z_d.detach(), dim=1),
        )
    z_c = outputs.get("z_c")
    style_z_c = style_outputs.get("z_c")
    if isinstance(z_c, Tensor) and isinstance(style_z_c, Tensor) and float(args.lambda_style_context_response) > 0.0:
        response = (style_z_c - z_c.detach()).norm(dim=1).mean()
        target_response = torch.as_tensor(
            float(args.style_context_response_target),
            device=response.device,
            dtype=response.dtype,
        )
        terms["style_context_response"] = torch.relu(target_response - response)
    return terms

def _lesion_effect_loss(
    reference_logits: Tensor,
    intervention_logits: Tensor,
    effect_mask: Tensor,
    margin: float,
    direction: str,
) -> Tensor:
    effect_mask = effect_mask.to(device=reference_logits.device, dtype=reference_logits.dtype).clamp(0.0, 1.0)
    if float(effect_mask.sum().detach().cpu()) <= 0.0:
        return reference_logits.sum() * 0.0
    reference_prob = torch.sigmoid(reference_logits.detach())
    intervention_prob = torch.sigmoid(intervention_logits)
    if direction == "paste":
        effect = intervention_prob - reference_prob
    elif direction == "erase":
        effect = reference_prob - intervention_prob
    else:
        raise ValueError(f"Unknown lesion intervention direction: {direction}")
    numerator = (effect * effect_mask).sum()
    denominator = effect_mask.sum().clamp_min(1.0)
    observed_effect = numerator / denominator
    return torch.relu(torch.as_tensor(float(margin), device=effect.device, dtype=effect.dtype) - observed_effect)


def _lesion_intervention_terms(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    intervention_outputs: dict[str, Tensor | tuple[Tensor, ...]],
    intervention_target: Tensor,
    effect_mask: Tensor,
    args: argparse.Namespace,
    prefix: str,
    direction: str,
) -> dict[str, Tensor]:
    terms: dict[str, Tensor] = {}
    logits = intervention_outputs.get("logits")
    factual_logits = outputs.get("logits")
    if isinstance(logits, Tensor):
        terms.update(_segmentation_terms(logits, intervention_target, args, f"{prefix}_seg"))
        if isinstance(factual_logits, Tensor) and float(args.lambda_lesion_intervention_effect) > 0.0:
            terms[f"{prefix}_effect"] = _lesion_effect_loss(
                factual_logits,
                logits,
                effect_mask,
                margin=float(args.lesion_effect_margin),
                direction=direction,
            )
    if float(args.lambda_lesion_mediator) > 0.0:
        spatial_attention = _spatial_disease_attention_loss(
            intervention_outputs.get("disease_attention_logits"),
            intervention_target,
        )
        spatial_region = _spatial_region_head_loss(
            intervention_outputs.get("spatial_region_logits"),
            intervention_target,
            args,
        )
        prototype = _prototype_mediator_loss(
            intervention_outputs.get("prototype_logits"),
            intervention_outputs.get("prototype_subregion_logits"),
            intervention_target,
            args,
        )
        boundary = _boundary_mediator_loss(
            intervention_outputs.get("boundary_logits"),
            intervention_target,
        )
        if spatial_attention is not None:
            terms[f"{prefix}_spatial_attention"] = spatial_attention
        if spatial_region is not None:
            terms[f"{prefix}_spatial_region"] = spatial_region
        if prototype is not None:
            terms[f"{prefix}_prototype_mediator"] = prototype
        if boundary is not None:
            terms[f"{prefix}_boundary_mediator"] = boundary
    return terms


def _feature_intervention_terms(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    masked_logits: Tensor,
    target: Tensor,
    args: argparse.Namespace,
) -> dict[str, Tensor]:
    terms: dict[str, Tensor] = {}
    if float(args.lambda_feature_intervention_seg) > 0.0:
        terms.update(_segmentation_terms(masked_logits, target, args, "feature_intervention_seg"))
    factual_logits = outputs.get("logits")
    if isinstance(factual_logits, Tensor) and float(args.lambda_feature_intervention_consistency) > 0.0:
        terms["feature_intervention_consistency"] = _probability_consistency_loss(masked_logits, factual_logits, args)
    return terms


def _registered_modality_terms(
    outputs: dict[str, Tensor | tuple[Tensor, ...]],
    registered_outputs: dict[str, Tensor | tuple[Tensor, ...]],
    target: Tensor,
    args: argparse.Namespace,
) -> dict[str, Tensor]:
    terms: dict[str, Tensor] = {}
    registered_logits = registered_outputs.get("logits")
    factual_logits = outputs.get("logits")
    if isinstance(registered_logits, Tensor) and float(args.lambda_registered_modality_seg) > 0.0:
        seg_terms = _segmentation_terms(
            registered_logits,
            target,
            _registered_segmentation_args(args),
            "registered_modality_seg",
        )
        registered_seg_weight = _registered_modality_small_lesion_weight(target, args)
        if isinstance(factual_logits, Tensor):
            registered_seg_weight = registered_seg_weight * _registered_modality_error_weight(
                factual_logits,
                target,
                args,
            )
        for name in ("registered_modality_seg", "registered_modality_seg_region"):
            if name in seg_terms:
                seg_terms[name] = seg_terms[name] * registered_seg_weight
        terms.update(seg_terms)
    if (
        isinstance(registered_logits, Tensor)
        and isinstance(factual_logits, Tensor)
        and float(getattr(args, "lambda_registered_modality_fusion_seg", 0.0)) > 0.0
    ):
        fusion_logits = _fuse_registered_modality_logits(
            factual_logits,
            registered_logits,
            str(getattr(args, "registered_modality_fusion_mode", "mean-probs")),
        )
        fusion_terms = _segmentation_terms(
            fusion_logits,
            target,
            _registered_segmentation_args(args),
            "registered_modality_fusion_seg",
        )
        fusion_seg_weight = _registered_modality_small_lesion_weight(target, args) * _registered_modality_error_weight(
            factual_logits,
            target,
            args,
        )
        for name in ("registered_modality_fusion_seg", "registered_modality_fusion_seg_region"):
            if name in fusion_terms:
                fusion_terms[name] = fusion_terms[name] * fusion_seg_weight
        terms.update(fusion_terms)
    if (
        isinstance(registered_logits, Tensor)
        and isinstance(factual_logits, Tensor)
        and float(args.lambda_registered_modality_consistency) > 0.0
    ):
        terms["registered_modality_consistency"] = _probability_consistency_loss(
            registered_logits,
            factual_logits,
            args,
        )
    if (
        isinstance(registered_logits, Tensor)
        and isinstance(factual_logits, Tensor)
        and float(getattr(args, "lambda_registered_modality_region_consistency", 0.0)) > 0.0
    ):
        weights = _parse_channel_weights(
            getattr(args, "registered_modality_region_consistency_weights", "1.0,1.5,2.5"),
            3,
            "--registered-modality-region-consistency-weights",
        )
        registered_regions = brats_region_probabilities(registered_logits)
        factual_regions = brats_region_probabilities(factual_logits.detach())
        region_delta = (registered_regions - factual_regions).pow(2)
        region_weights = torch.as_tensor(weights, device=region_delta.device, dtype=region_delta.dtype)
        terms["registered_modality_region_consistency"] = (
            region_delta * region_weights.view(1, 3, 1, 1, 1)
        ).sum(dim=1).div(region_weights.sum().clamp_min(1e-6)).mean()
    if (
        isinstance(registered_logits, Tensor)
        and isinstance(factual_logits, Tensor)
        and float(getattr(args, "lambda_registered_modality_wt_consistency", 0.0)) > 0.0
    ):
        registered_wt = brats_region_probabilities(registered_logits)[:, :1]
        factual_wt = brats_region_probabilities(factual_logits.detach())[:, :1]
        dims = tuple(range(2, registered_wt.ndim))
        intersection = (registered_wt * factual_wt).sum(dim=dims)
        denominator = registered_wt.sum(dim=dims) + factual_wt.sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        terms["registered_modality_wt_consistency"] = (1.0 - dice).mean()
    if (
        isinstance(registered_logits, Tensor)
        and isinstance(factual_logits, Tensor)
        and float(getattr(args, "lambda_registered_modality_view_advantage_distillation", 0.0)) > 0.0
    ):
        terms["registered_modality_view_advantage_distillation"] = (
            _registered_modality_view_advantage_distillation_loss(
                factual_logits,
                registered_logits,
                target,
                args,
            )
        )
    z_d = outputs.get("z_d")
    registered_z_d = registered_outputs.get("z_d")
    if (
        isinstance(z_d, Tensor)
        and isinstance(registered_z_d, Tensor)
        and float(args.lambda_registered_modality_disease_invariance) > 0.0
    ):
        terms["registered_modality_disease_invariance"] = F.mse_loss(
            F.normalize(registered_z_d, dim=1),
            F.normalize(z_d.detach(), dim=1),
        )
    return terms


def _registered_modality_small_lesion_weight(target: Tensor, args: argparse.Namespace) -> Tensor:
    emphasis = float(getattr(args, "registered_modality_small_lesion_emphasis", 0.0) or 0.0)
    return _small_lesion_emphasis_weights(
        target,
        emphasis=emphasis,
        reference_fractions=getattr(args, "registered_modality_small_lesion_reference_fractions", "0.015,0.006,0.003"),
        region_weights=getattr(args, "registered_modality_small_lesion_region_weights", "0.5,1.0,1.5"),
        max_weight=float(getattr(args, "registered_modality_small_lesion_max_weight", 2.0) or 2.0),
        reference_name="--registered-modality-small-lesion-reference-fractions",
        region_weights_name="--registered-modality-small-lesion-region-weights",
    ).mean()


def _registered_modality_error_weight(factual_logits: Tensor, target: Tensor, args: argparse.Namespace) -> Tensor:
    emphasis = float(getattr(args, "registered_modality_error_emphasis", 0.0) or 0.0)
    if emphasis <= 0.0:
        return torch.ones((), device=factual_logits.device, dtype=factual_logits.dtype)
    weights = _parse_channel_weights(
        getattr(args, "registered_modality_error_region_weights", "0.3,1.0,1.5"),
        3,
        "--registered-modality-error-region-weights",
    )
    region_probs = brats_region_probabilities(factual_logits.detach())
    region_target = brats_region_targets(target).to(device=region_probs.device, dtype=region_probs.dtype)
    dims = tuple(range(2, region_probs.ndim))
    intersection = (region_probs * region_target).sum(dim=dims)
    denominator = region_probs.sum(dim=dims) + region_target.sum(dim=dims)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    present = (region_target.sum(dim=dims) > 0.0).to(dtype=region_probs.dtype)
    region_weight = torch.as_tensor(weights, device=region_probs.device, dtype=region_probs.dtype).clamp_min(0.0)
    numerator = ((1.0 - dice).clamp(0.0, 1.0) * present * region_weight.view(1, 3)).sum(dim=1)
    weight_sum = (present * region_weight.view(1, 3)).sum(dim=1).clamp_min(1e-6)
    error_score = (numerator / weight_sum).mean()
    max_weight = max(1.0, float(getattr(args, "registered_modality_error_max_weight", 1.5) or 1.5))
    return (1.0 + emphasis * error_score).clamp(1.0, max_weight).detach()


def _soft_region_dice(region_probs: Tensor, region_target: Tensor, eps: float = 1e-6) -> Tensor:
    dims = tuple(range(2, region_probs.ndim))
    intersection = (region_probs * region_target).sum(dim=dims)
    denominator = region_probs.sum(dim=dims) + region_target.sum(dim=dims)
    return (2.0 * intersection + eps) / (denominator + eps)


def _registered_modality_view_advantage_distillation_loss(
    factual_logits: Tensor,
    registered_logits: Tensor,
    target: Tensor,
    args: argparse.Namespace,
) -> Tensor:
    weights = _parse_channel_weights(
        getattr(args, "registered_modality_view_advantage_region_weights", "0.25,1.0,1.5"),
        3,
        "--registered-modality-view-advantage-region-weights",
    )
    margin = max(0.0, float(getattr(args, "registered_modality_view_advantage_margin", 0.0) or 0.0))
    factual_regions = brats_region_probabilities(factual_logits)
    registered_regions = brats_region_probabilities(registered_logits)
    region_target = brats_region_targets(target).to(device=factual_regions.device, dtype=factual_regions.dtype)
    factual_dice = _soft_region_dice(factual_regions.detach(), region_target)
    registered_dice = _soft_region_dice(registered_regions.detach(), region_target)
    factual_better = (factual_dice > registered_dice + margin).to(dtype=factual_regions.dtype)
    registered_better = (registered_dice > factual_dice + margin).to(dtype=factual_regions.dtype)
    dims = tuple(range(2, factual_regions.ndim))
    factual_student_loss = (factual_regions - registered_regions.detach()).square().mean(dim=dims)
    registered_student_loss = (registered_regions - factual_regions.detach()).square().mean(dim=dims)
    per_region_loss = registered_better * factual_student_loss + factual_better * registered_student_loss
    region_weights = torch.as_tensor(weights, device=factual_regions.device, dtype=factual_regions.dtype).clamp_min(0.0)
    active_weights = (registered_better + factual_better).clamp_max(1.0) * region_weights.view(1, 3)
    numerator = (per_region_loss * region_weights.view(1, 3)).sum(dim=1)
    denominator = active_weights.sum(dim=1).clamp_min(1e-6)
    return (numerator / denominator).mean()


def _registered_segmentation_args(args: argparse.Namespace) -> argparse.Namespace:
    channel_weights = _scheduled_registered_weight_spec(
        args,
        target_attr="registered_modality_channel_loss_weights",
        start_attr="registered_modality_start_channel_loss_weights",
        base_attr="channel_loss_weights",
        count=3,
        name="--registered-modality-channel-loss-weights",
    )
    region_weights = _scheduled_registered_weight_spec(
        args,
        target_attr="registered_modality_region_loss_weights",
        start_attr="registered_modality_start_region_loss_weights",
        base_attr="region_loss_weights",
        count=3,
        name="--registered-modality-region-loss-weights",
    )
    if not channel_weights and not region_weights:
        return args
    registered_args = copy.copy(args)
    if channel_weights:
        setattr(registered_args, "channel_loss_weights", channel_weights)
    if region_weights:
        setattr(registered_args, "region_loss_weights", region_weights)
    return registered_args


def _registered_modality_weight_schedule_alpha(args: argparse.Namespace, epoch: int | None = None) -> float:
    has_scheduled_target = bool(
        str(getattr(args, "registered_modality_channel_loss_weights", "") or "").strip()
        or str(getattr(args, "registered_modality_region_loss_weights", "") or "").strip()
    )
    if not has_scheduled_target:
        return 0.0
    ramp_epochs = int(getattr(args, "registered_modality_weight_ramp_epochs", 0) or 0)
    if ramp_epochs <= 0:
        return 1.0
    current_epoch = int(epoch if epoch is not None else getattr(args, "_current_epoch", 1))
    if ramp_epochs <= 1:
        return 1.0
    return float(min(max((current_epoch - 1) / max(ramp_epochs - 1, 1), 0.0), 1.0))


def _format_weight_spec(weights: tuple[float, ...]) -> str:
    return ",".join(f"{weight:.6g}" for weight in weights)


def _scheduled_registered_weight_spec(
    args: argparse.Namespace,
    *,
    target_attr: str,
    start_attr: str,
    base_attr: str,
    count: int,
    name: str,
) -> str:
    target_spec = str(getattr(args, target_attr, "") or "").strip()
    if not target_spec:
        return ""
    ramp_epochs = int(getattr(args, "registered_modality_weight_ramp_epochs", 0) or 0)
    if ramp_epochs <= 0:
        return target_spec
    target = _parse_channel_weights(target_spec, count, name)
    start_spec = str(getattr(args, start_attr, "") or "").strip()
    if not start_spec:
        start_spec = str(getattr(args, base_attr, "1.0,1.0,1.0") or "")
    start = _parse_channel_weights(start_spec, count, f"--{start_attr.replace('_', '-')}")
    alpha = _registered_modality_weight_schedule_alpha(args)
    scheduled = tuple((1.0 - alpha) * left + alpha * right for left, right in zip(start, target, strict=True))
    return _format_weight_spec(scheduled)


def _logit(probability: Tensor) -> Tensor:
    return torch.logit(probability.clamp(1e-4, 1.0 - 1e-4))


def _fuse_registered_modality_logits(native_logits: Tensor, registered_logits: Tensor, fusion: str) -> Tensor:
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
    raise ValueError(f"Unsupported registered modality fusion: {fusion}")


def _registered_modality_loss_weight(name: str, args: argparse.Namespace) -> float:
    if name in {"registered_modality_seg", "registered_modality_seg_region"}:
        return float(getattr(args, "lambda_registered_modality_seg", 0.0))
    if name in {"registered_modality_fusion_seg", "registered_modality_fusion_seg_region"}:
        return float(getattr(args, "lambda_registered_modality_fusion_seg", 0.0))
    if name == "registered_modality_consistency":
        return float(getattr(args, "lambda_registered_modality_consistency", 0.0))
    if name == "registered_modality_region_consistency":
        return float(getattr(args, "lambda_registered_modality_region_consistency", 0.0))
    if name == "registered_modality_wt_consistency":
        return float(getattr(args, "lambda_registered_modality_wt_consistency", 0.0))
    if name == "registered_modality_view_advantage_distillation":
        return float(getattr(args, "lambda_registered_modality_view_advantage_distillation", 0.0))
    if name == "registered_modality_disease_invariance":
        return float(getattr(args, "lambda_registered_modality_disease_invariance", 0.0))
    return 0.0


def _weighted_total(terms: dict[str, Tensor], args: argparse.Namespace) -> Tensor:
    registered_terms = {
        name: value for name, value in terms.items() if name.startswith("registered_modality_")
    }
    base_terms = {
        name: value for name, value in terms.items() if name not in registered_terms
    }
    first_value = next(iter(terms.values()))
    total = _shared_weighted_total(base_terms, args) if base_terms else torch.zeros((), device=first_value.device)
    for name, value in registered_terms.items():
        total = total + _registered_modality_loss_weight(name, args) * value
    return total


@torch.no_grad()
def prefill_lesion_intervention_bank(
    bank: LesionInterventionBank | None,
    loader: DataLoader,
    max_batches: int | None,
) -> dict[str, float]:
    if bank is None:
        return {}
    if max_batches is not None and max_batches <= 0:
        return {"lesion_bank/prefill_patches": float(len(bank))}
    for batch_idx, batch in enumerate(tqdm(loader, desc="lesion-bank-prefill", leave=False), start=1):
        bank.update(batch["image"], batch["mask"])
        if len(bank) >= bank.max_patches:
            break
        if max_batches is not None and batch_idx >= max_batches:
            break
    return {"lesion_bank/prefill_patches": float(len(bank))}


@torch.no_grad()
def build_category_confounder_dictionary(
    model: CausalMedNeXt,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Estimate category confounders c_k = E[f(v) | Y_k(v)=1] from training masks."""
    if float(getattr(model, "category_confounder_scale", 0.0)) <= 0.0:
        return {}
    model.eval()
    model.reset_category_confounders()
    channels = int(model.category_confounders.shape[1])
    classes = int(model.category_confounders.shape[0])
    sums = torch.zeros(classes, channels, device=device)
    counts = torch.zeros(classes, device=device)
    for batch_idx, batch in enumerate(tqdm(loader, desc="category-confounders", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        features = model.encode_features(image)
        z_d, z_c, _ = model.encode_sdd_latents(features)
        outputs = model._segment_outputs_from_latents(features, z_d, z_c)
        high_res = outputs["causal_high_res_features"]
        if not isinstance(high_res, Tensor):
            continue
        if tuple(target.shape[-3:]) != tuple(high_res.shape[-3:]):
            target = F.interpolate(target.float(), size=high_res.shape[-3:], mode="nearest")
        for class_idx in range(min(classes, target.shape[1])):
            mask = (target[:, class_idx : class_idx + 1] >= float(threshold)).to(dtype=high_res.dtype)
            count = mask.sum()
            if float(count.detach().cpu()) <= 0.0:
                continue
            sums[class_idx] += (high_res * mask).sum(dim=(0, 2, 3, 4))
            counts[class_idx] += count
        if max_batches is not None and batch_idx >= max_batches:
            break
    confounders = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1.0)[:, None], sums)
    model.set_category_confounders(confounders.detach(), counts.detach())
    return {
        f"category_confounder/count_{idx}": float(counts[idx].detach().cpu())
        for idx in range(classes)
    }


@torch.no_grad()
def build_sdd_cite_bank(
    model: CausalMedNeXt,
    loader: DataLoader,
    device: torch.device,
    max_contexts: int,
    max_batches: int | None = None,
    sampling: str = "uniform",
    seed: int = 7,
) -> dict[str, Tensor] | None:
    model.eval()
    z_d_chunks: list[Tensor] = []
    z_c_chunks: list[Tensor] = []
    z_t_chunks: list[Tensor] = []
    propensity_chunks: list[Tensor] = []
    treatment_label_chunks: list[Tensor] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="sdd-cite-bank", leave=False), start=1):
        image = batch["image"].to(device)
        features = model.encode_features(image)
        z_d, z_c, z_t = model.encode_sdd_latents(features)
        propensity = model.treatment_propensity(z_t, z_c)
        if propensity is None:
            propensity = torch.full((z_c.shape[0],), 0.5, device=z_c.device, dtype=z_c.dtype)
        z_d_chunks.append(z_d.detach().cpu())
        z_c_chunks.append(z_c.detach().cpu())
        z_t_chunks.append(z_t.detach().cpu())
        propensity_chunks.append(propensity.detach().cpu())
        treatment_label = batch.get("observed_treatment_label")
        if isinstance(treatment_label, Tensor):
            treatment_label_chunks.append(treatment_label.detach().cpu().view(-1).long())
        else:
            treatment = batch.get("observed_treatment")
            if isinstance(treatment, Tensor):
                if treatment.ndim == 2 and treatment.shape[1] > 1:
                    treatment_label_chunks.append(treatment.detach().cpu().argmax(dim=1).long())
                else:
                    treatment_label_chunks.append((treatment.detach().cpu().view(-1) > 0.5).long())
        if max_batches is not None and batch_idx >= max_batches:
            break
    if not z_c_chunks:
        return None
    z_d_bank = torch.cat(z_d_chunks, dim=0)
    z_c_bank = torch.cat(z_c_chunks, dim=0)
    z_t_bank = torch.cat(z_t_chunks, dim=0)
    propensity_bank = torch.cat(propensity_chunks, dim=0).view(-1)
    sampled_z_c = _subsample_context_bank(z_c_bank, max_contexts=max_contexts, strategy=sampling, seed=seed)
    if sampled_z_c.shape[0] == z_c_bank.shape[0] and torch.equal(sampled_z_c, z_c_bank):
        indices = torch.arange(z_c_bank.shape[0])
    else:
        distances = torch.cdist(sampled_z_c, z_c_bank)
        indices = distances.argmin(dim=1)
    bank = {
        "z_d": z_d_bank[indices],
        "z_c": z_c_bank[indices],
        "z_t": z_t_bank[indices],
        "propensity": propensity_bank[indices],
    }
    if treatment_label_chunks:
        treatment_label_bank = torch.cat(treatment_label_chunks, dim=0)
        bank["treatment_label"] = treatment_label_bank[indices]
    return bank


def _run_train_epoch(
    model: CausalMedNeXt,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    context_bank: Tensor | None,
    contrastive_bank: dict[str, Tensor] | None,
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
    lesion_bank: LesionInterventionBank | None = None,
    teacher_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    loss_logs: list[dict[str, float]] = []
    metric_items: list[dict[str, float]] = []
    bank_device = context_bank.to(device) if context_bank is not None else None
    for batch_idx, batch in enumerate(tqdm(loader, desc="mednext-causal-train", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            image,
            context_bank=bank_device,
            max_adjustment_contexts=args.adjustment_contexts,
            adjustment_context_selection=args.adjustment_context_selection,
            adversary_strength=args.adversary_strength,
            max_contrastive_negatives=args.cite_bank_negatives,
        )
        model.add_cite_outputs(outputs, contrastive_bank, max_negatives=args.cite_bank_negatives)
        _add_context_swap_outputs(model, outputs, bank_device, args)
        teacher_logits = _teacher_logits(teacher_model, image)
        if teacher_logits is not None:
            outputs["teacher_logits"] = teacher_logits
        terms = _causal_loss_terms(outputs, batch, args, proxy_layout)
        registered_outputs = None
        registered_image = batch.get("registered_image")
        if isinstance(registered_image, Tensor) and _uses_registered_modality_training(args):
            registered_outputs = model(
                registered_image.to(device),
                context_bank=None,
                adversary_strength=args.adversary_strength,
                max_contrastive_negatives=args.cite_bank_negatives,
            )
            terms.update(_registered_modality_terms(outputs, registered_outputs, target, args))
        if _should_apply_style_intervention(args):
            style_image = apply_style_intervention(image, args)
            style_outputs = model(
                style_image,
                context_bank=None,
                adversary_strength=args.adversary_strength,
                max_contrastive_negatives=args.cite_bank_negatives,
            )
            terms.update(_style_intervention_terms(outputs, style_outputs, target, args))
        if _should_apply_feature_intervention(args):
            features = outputs.get("features")
            z_d = outputs.get("z_d")
            z_c = outputs.get("z_c")
            if isinstance(features, tuple) and isinstance(z_d, Tensor) and isinstance(z_c, Tensor):
                masked_features = _apply_causal_feature_mask(features, args)
                masked_logits = model.segment_from_latents(masked_features, z_d, z_c)
                terms.update(_feature_intervention_terms(outputs, masked_logits, target, args))
        if lesion_bank is not None and _should_apply_lesion_intervention(args):
            paste = lesion_bank.paste(image, target)
            if paste is not None and float(args.lambda_lesion_paste_seg) > 0.0:
                paste_image, paste_target, paste_mask = paste
                paste_outputs = model(
                    paste_image,
                    context_bank=None,
                    adversary_strength=args.adversary_strength,
                    max_contrastive_negatives=args.cite_bank_negatives,
                )
                terms.update(
                    _lesion_intervention_terms(
                        outputs,
                        paste_outputs,
                        paste_target,
                        paste_mask,
                        args,
                        prefix="lesion_paste",
                        direction="paste",
                    )
                )
            erase = lesion_bank.erase(image, target)
            if erase is not None and float(args.lambda_lesion_erase_seg) > 0.0:
                erase_image, erase_target, erase_mask = erase
                erase_outputs = model(
                    erase_image,
                    context_bank=None,
                    adversary_strength=args.adversary_strength,
                    max_contrastive_negatives=args.cite_bank_negatives,
                )
                terms.update(
                    _lesion_intervention_terms(
                        outputs,
                        erase_outputs,
                        erase_target,
                        erase_mask,
                        args,
                        prefix="lesion_erase",
                        direction="erase",
                    )
                )
        total = _weighted_total(terms, args)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        item = {"loss/total": float(total.detach().cpu())}
        item.update(_float_terms(terms))
        loss_logs.append(item)
        metrics = brats_region_metrics(outputs["logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold)
        if "adjusted_logits" in outputs:
            metrics.update(_prefix_metrics(brats_region_metrics(outputs["adjusted_logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold), "adjusted"))
        if "context_swap_logits" in outputs:
            metrics.update(_prefix_metrics(brats_region_metrics(outputs["context_swap_logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold), "context_swap"))
        if registered_outputs is not None and isinstance(registered_outputs.get("logits"), Tensor):
            metrics.update(
                _prefix_metrics(
                    brats_region_metrics(registered_outputs["logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold),
                    "registered_modality",
                )
            )
        metrics.update(_et_volume_veto_metric_item(outputs))
        metric_items.append(metrics)
        if lesion_bank is not None:
            lesion_bank.update(image, target)
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
    return {**_average_metric_dicts(loss_logs), **_average_metric_dicts(metric_items)}


@torch.no_grad()
def _run_eval_epoch(
    model: CausalMedNeXt,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    context_bank: Tensor | None,
    contrastive_bank: dict[str, Tensor] | None,
    proxy_layout: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, float]:
    model.eval()
    loss_logs: list[dict[str, float]] = []
    metric_items: list[dict[str, float]] = []
    adjusted_metric_items: list[dict[str, float]] = []
    calibration_candidates = parse_threshold_candidates(getattr(args, "checkpoint_calibration_thresholds", None))
    calibration_sweep = (
        BratsRegionThresholdSweep(
            calibration_candidates,
            objective=getattr(args, "checkpoint_calibration_objective", "mean"),
        )
        if calibration_candidates
        else None
    )
    adjusted_calibration_sweep = (
        BratsRegionThresholdSweep(
            calibration_candidates,
            objective=getattr(args, "checkpoint_calibration_objective", "mean"),
        )
        if calibration_candidates
        else None
    )
    registered_tta_calibration_sweep = (
        BratsRegionThresholdSweep(
            calibration_candidates,
            objective=getattr(args, "checkpoint_calibration_objective", "mean"),
        )
        if calibration_candidates
        else None
    )
    bank_device = context_bank.to(device) if context_bank is not None else None
    for batch_idx, batch in enumerate(tqdm(loader, desc="mednext-causal-val", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        outputs = model(
            image,
            context_bank=bank_device,
            max_adjustment_contexts=args.adjustment_contexts,
            adjustment_context_selection=args.adjustment_context_selection,
            adversary_strength=args.adversary_strength,
            max_contrastive_negatives=args.cite_bank_negatives,
        )
        model.add_cite_outputs(outputs, contrastive_bank, max_negatives=args.cite_bank_negatives)
        _add_context_swap_outputs(model, outputs, bank_device, args)
        terms = _causal_loss_terms(outputs, batch, args, proxy_layout)
        total = _weighted_total(terms, args)

        item = {"loss/total": float(total.detach().cpu())}
        item.update(_float_terms(terms))
        loss_logs.append(item)
        logits_cpu = outputs["logits"].detach().cpu()
        target_cpu = target.detach().cpu()
        metrics = brats_region_metrics(logits_cpu, target_cpu, threshold=args.threshold)
        if calibration_sweep is not None:
            calibration_sweep.update(logits_cpu, target_cpu)
        if "adjusted_logits" in outputs:
            adjusted_cpu = outputs["adjusted_logits"].detach().cpu()
            adjusted_metrics = brats_region_metrics(adjusted_cpu, target_cpu, threshold=args.threshold)
            adjusted_metric_items.append(adjusted_metrics)
            metrics.update(_prefix_metrics(adjusted_metrics, "adjusted"))
            if adjusted_calibration_sweep is not None:
                adjusted_calibration_sweep.update(adjusted_cpu, target_cpu)
        if "context_swap_logits" in outputs:
            metrics.update(_prefix_metrics(brats_region_metrics(outputs["context_swap_logits"].detach().cpu(), target_cpu, threshold=args.threshold), "context_swap"))
        registered_image = batch.get("registered_image")
        if isinstance(registered_image, Tensor) and _uses_registered_modality_validation(args):
            registered_outputs = model(
                registered_image.to(device),
                context_bank=None,
                adversary_strength=args.adversary_strength,
                max_contrastive_negatives=args.cite_bank_negatives,
            )
            registered_logits = registered_outputs.get("logits")
            if isinstance(registered_logits, Tensor):
                registered_tta_logits = _fuse_registered_modality_logits(
                    outputs["logits"],
                    registered_logits,
                    str(getattr(args, "checkpoint_registered_modality_fusion", "mean-probs")),
                )
                metrics.update(
                    registered_modality_consistency_metrics(
                        outputs["logits"],
                        registered_logits,
                        fused_logits=registered_tta_logits,
                        threshold=float(args.threshold),
                    )
                )
                registered_tta_cpu = registered_tta_logits.detach().cpu()
                metrics.update(
                    _prefix_metrics(
                        brats_region_metrics(registered_tta_cpu, target_cpu, threshold=args.threshold),
                        "registered_tta",
                    )
                )
                if registered_tta_calibration_sweep is not None:
                    registered_tta_calibration_sweep.update(registered_tta_cpu, target_cpu)
        metrics.update(_et_volume_veto_metric_item(outputs))
        metric_items.append(metrics)
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break
    summary = {**_average_metric_dicts(loss_logs), **_average_metric_dicts(metric_items)}
    if calibration_sweep is not None:
        summary.update(prefix_metrics(calibration_sweep.summary(), "sweep_region_calibrated"))
    if adjusted_metric_items and adjusted_calibration_sweep is not None:
        summary.update(prefix_metrics(adjusted_calibration_sweep.summary(), "adjusted_sweep_region_calibrated"))
    if registered_tta_calibration_sweep is not None:
        summary.update(prefix_metrics(registered_tta_calibration_sweep.summary(), "registered_tta_sweep_region_calibrated"))
    if "loss/total" in summary:
        summary["selection/negative_loss"] = -float(summary["loss/total"])
    return summary


def _add_causal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--init-checkpoint", help="Optional causal MedNeXt checkpoint to initialize compatible weights before training.")
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument(
        "--adjustment-context-selection",
        choices=["uniform", "nearest", "farthest", "diverse-nearest"],
        default="uniform",
        help="How the SCM adjustment selects contexts from the proxy bank for each target case.",
    )
    parser.add_argument("--context-swap-strategy", choices=["none", "random", "nearest", "farthest"], default="none")
    parser.add_argument("--context-bank-refresh-epochs", type=int, default=1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-calibration-thresholds", help="Optional WT/TC/ET threshold grid for calibrated validation checkpoint selection.")
    parser.add_argument(
        "--checkpoint-calibration-objective",
        choices=CALIBRATION_OBJECTIVES,
        default="mean",
        help="Objective used when choosing validation thresholds from --checkpoint-calibration-thresholds.",
    )
    parser.add_argument("--checkpoint-registered-modality-tta", action="store_true", help="Use native/ANTS registered-modality TTA metrics for validation checkpoint selection.")
    parser.add_argument(
        "--checkpoint-registered-modality-selector",
        choices=["dice", "agreement", "region-prob-similarity", "stability", "prob-response", "region-prob-response", "val-loss"],
        default="dice",
        help="Checkpoint selector to use when --checkpoint-registered-modality-tta is enabled.",
    )
    parser.add_argument(
        "--checkpoint-registered-modality-fusion",
        choices=["mean-logits", "mean-probs", "max-probs", "registered-only"],
        default="mean-probs",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-stability-margin", type=float, default=0.03)
    parser.add_argument("--context-response-target", type=float, default=0.0)
    parser.add_argument("--adversary-strength", type=float, default=1.0)
    parser.add_argument("--modulation-scale", type=float, default=0.1)
    parser.add_argument("--causal-residual-scale", type=float, default=0.2)
    parser.add_argument("--spatial-refiner-scale", type=float, default=0.5)
    parser.add_argument("--region-fusion-scale", type=float, default=0.0)
    parser.add_argument("--prototype-dim", type=int, default=32)
    parser.add_argument("--prototype-fusion-scale", type=float, default=0.0)
    parser.add_argument("--prototype-temperature", type=float, default=0.1)
    parser.add_argument("--category-confounder-scale", type=float, default=0.0)
    parser.add_argument("--category-confounder-temperature", type=float, default=0.2)
    parser.add_argument("--modality-prior-scale", type=float, default=0.0)
    parser.add_argument("--logit-calibration-scale", type=float, default=0.0)
    parser.add_argument("--cascade-refiner-scale", type=float, default=0.0)
    parser.add_argument("--frontdoor-mediator-scale", type=float, default=0.0)
    parser.add_argument("--frontdoor-residual-scale", type=float, default=0.25)
    parser.add_argument("--use-causal-mediator-router", action="store_true")
    parser.add_argument("--use-nested-causal-intervention", action="store_true")
    parser.add_argument("--nested-causal-gate-scale", type=float, default=1.0)
    parser.add_argument("--region-causal-bottleneck-scale", type=float, default=0.0)
    parser.add_argument("--region-causal-background-leak", type=float, default=0.05)
    parser.add_argument("--region-causal-base", choices=["prior", "factual"], default="prior")
    parser.add_argument("--region-causal-mask-source", choices=["spatial", "factual"], default="spatial")
    parser.add_argument("--max-category-confounder-batches", type=int)
    parser.add_argument("--contrastive-dim", type=int, default=64)
    parser.add_argument("--cite-temperature", type=float, default=0.2)
    parser.add_argument("--cite-bank-negatives", type=int, default=16)
    parser.add_argument("--region-volume-scale", type=float, default=1000.0)
    parser.add_argument("--seg-loss-mode", choices=["bce_dice", "balanced_focal"], default="bce_dice")
    parser.add_argument("--channel-loss-weights", default="1.0,1.0,1.0")
    parser.add_argument("--region-loss-weights", default="1.0,1.5,2.5")
    parser.add_argument("--balanced-bce-max-pos-weight", type=float, default=50.0)
    parser.add_argument("--focal-tversky-alpha", type=float, default=0.7)
    parser.add_argument("--focal-tversky-beta", type=float, default=0.3)
    parser.add_argument("--focal-tversky-gamma", type=float, default=0.75)
    parser.add_argument("--lambda-volume-prior-loss", type=float, default=0.0)
    parser.add_argument("--volume-prior-scale", type=float, default=1000.0)
    parser.add_argument("--distill-channel-weights", default="1.0,1.0,2.0")
    parser.add_argument("--proxy-loss-mode", choices=["mse", "typed"], default="typed")
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-region-loss", type=float, default=0.0)
    parser.add_argument("--lambda-adjustment", type=float, default=0.25)
    parser.add_argument("--lambda-context-swap", type=float, default=0.0)
    parser.add_argument("--lambda-context-swap-region", type=float, default=0.0)
    parser.add_argument("--lambda-context-swap-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-teacher-distill", type=float, default=0.0)
    parser.add_argument("--lambda-adjusted-teacher-distill", type=float, default=0.0)
    parser.add_argument("--lambda-context-swap-teacher-distill", type=float, default=0.0)
    parser.add_argument("--lambda-context-stability", type=float, default=0.02)
    parser.add_argument("--lambda-context-response", type=float, default=0.0)
    parser.add_argument("--lambda-context-proxy", type=float, default=0.05)
    parser.add_argument("--lambda-disease-proxy", type=float, default=0.05)
    parser.add_argument("--lambda-annotation-proxy", type=float, default=0.01)
    parser.add_argument("--lambda-context-from-disease-adversary", type=float, default=0.02)
    parser.add_argument("--lambda-disease-from-context-adversary", type=float, default=0.02)
    parser.add_argument("--lambda-region-volume-proxy", type=float, default=0.05)
    parser.add_argument("--et-volume-veto-scale", type=float, default=0.0)
    parser.add_argument("--et-volume-veto-warmup-epochs", type=int, default=0)
    parser.add_argument("--et-volume-veto-ramp-epochs", type=int, default=0)
    parser.add_argument("--et-volume-veto-multiplier", type=float, default=4.0)
    parser.add_argument("--et-volume-veto-min-fraction", type=float, default=5e-4)
    parser.add_argument("--et-volume-veto-max-bias", type=float, default=4.0)
    parser.add_argument("--lambda-region-from-context-adversary", type=float, default=0.02)
    parser.add_argument("--lambda-sdd-context-teacher", type=float, default=0.03)
    parser.add_argument("--lambda-sdd-region-teacher", type=float, default=0.05)
    parser.add_argument("--lambda-sdd-context-distill", type=float, default=0.02)
    parser.add_argument("--lambda-sdd-region-distill", type=float, default=0.03)
    parser.add_argument("--lambda-sdd-treatment", type=float, default=0.05)
    parser.add_argument("--lambda-sdd-treatment-disentangle", type=float, default=0.02)
    parser.add_argument("--lambda-sdd-outcome", type=float, default=0.05)
    parser.add_argument("--lambda-sdd-outcome-disentangle", type=float, default=0.03)
    parser.add_argument("--lambda-sdd-imbalance", type=float, default=0.01)
    parser.add_argument("--lambda-cite-contrastive", type=float, default=0.05)
    parser.add_argument("--lambda-spatial-disease-attention", type=float, default=0.0)
    parser.add_argument("--lambda-spatial-region-head", type=float, default=0.0)
    parser.add_argument("--lambda-subregion-prior", type=float, default=0.0)
    parser.add_argument("--lambda-prototype-mediator", type=float, default=0.0)
    parser.add_argument("--lambda-boundary-mediator", type=float, default=0.0)
    parser.add_argument("--lambda-category-confounder", type=float, default=0.0)
    parser.add_argument("--lambda-modality-prior", type=float, default=0.0)
    parser.add_argument("--lambda-cascade-error", type=float, default=0.0)
    parser.add_argument("--lambda-frontdoor-region", type=float, default=0.0)
    parser.add_argument("--lambda-frontdoor-subregion", type=float, default=0.0)
    parser.add_argument("--lambda-frontdoor-logits", type=float, default=0.0)
    parser.add_argument("--lambda-frontdoor-balanced-mediator", type=float, default=0.0)
    parser.add_argument("--lambda-frontdoor-router", type=float, default=0.0)
    parser.add_argument("--lambda-region-causal-logits", type=float, default=0.0)
    parser.add_argument("--lambda-region-causal-balanced", type=float, default=0.0)
    parser.add_argument("--lambda-region-causal-teacher-distill", type=float, default=0.0)
    parser.add_argument("--lambda-nested-causal-region", type=float, default=0.0)
    parser.add_argument("--lambda-nested-causal-subregion", type=float, default=0.0)
    parser.add_argument("--lambda-nested-causal-balanced", type=float, default=0.0)
    parser.add_argument("--lambda-nested-causal-router", type=float, default=0.0)
    parser.add_argument("--lambda-causal-refiner-sparsity", type=float, default=0.0)
    parser.add_argument("--style-intervention-prob", type=float, default=0.0)
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
    parser.add_argument("--style-context-response-target", type=float, default=1.0)
    parser.add_argument("--lambda-style-intervention-seg", type=float, default=0.0)
    parser.add_argument("--lambda-style-intervention-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-style-disease-invariance", type=float, default=0.0)
    parser.add_argument("--lambda-style-context-response", type=float, default=0.0)
    parser.add_argument("--lambda-registered-modality-seg", type=float, default=0.0)
    parser.add_argument("--registered-modality-channel-loss-weights", default="")
    parser.add_argument("--registered-modality-region-loss-weights", default="")
    parser.add_argument("--registered-modality-start-channel-loss-weights", default="")
    parser.add_argument("--registered-modality-start-region-loss-weights", default="")
    parser.add_argument("--registered-modality-weight-ramp-epochs", type=int, default=0)
    parser.add_argument("--registered-modality-small-lesion-emphasis", type=float, default=0.0)
    parser.add_argument("--registered-modality-small-lesion-reference-fractions", default="0.015,0.006,0.003")
    parser.add_argument("--registered-modality-small-lesion-region-weights", default="0.5,1.0,1.5")
    parser.add_argument("--registered-modality-small-lesion-max-weight", type=float, default=2.0)
    parser.add_argument("--registered-modality-error-emphasis", type=float, default=0.0)
    parser.add_argument("--registered-modality-error-region-weights", default="0.3,1.0,1.5")
    parser.add_argument("--registered-modality-error-max-weight", type=float, default=1.5)
    parser.add_argument("--hard-case-sampler-emphasis", type=float, default=0.0)
    parser.add_argument("--hard-case-sampler-reference-fractions", default="0.015,0.006,0.003")
    parser.add_argument("--hard-case-sampler-region-weights", default="0.5,1.0,1.5")
    parser.add_argument("--hard-case-sampler-max-weight", type=float, default=3.0)
    parser.add_argument("--hard-case-sampler-epoch-multiplier", type=float, default=1.0)
    parser.add_argument("--lambda-registered-modality-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-registered-modality-region-consistency", type=float, default=0.0)
    parser.add_argument("--registered-modality-region-consistency-weights", default="1.0,1.5,2.5")
    parser.add_argument("--lambda-registered-modality-wt-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-registered-modality-fusion-seg", type=float, default=0.0)
    parser.add_argument(
        "--registered-modality-fusion-mode",
        choices=["mean-probs", "mean-logits", "max-probs", "registered-only"],
        default="mean-probs",
    )
    parser.add_argument("--lambda-registered-modality-view-advantage-distillation", type=float, default=0.0)
    parser.add_argument("--registered-modality-view-advantage-region-weights", default="0.25,1.0,1.5")
    parser.add_argument("--registered-modality-view-advantage-margin", type=float, default=0.0)
    parser.add_argument("--lambda-registered-modality-disease-invariance", type=float, default=0.0)
    parser.add_argument("--feature-intervention-prob", type=float, default=0.0)
    parser.add_argument("--feature-mask-prob", type=float, default=0.15)
    parser.add_argument("--feature-mask-block-size", type=int, default=4)
    parser.add_argument("--lambda-feature-intervention-seg", type=float, default=0.0)
    parser.add_argument("--lambda-feature-intervention-consistency", type=float, default=0.0)
    parser.add_argument("--lesion-intervention-prob", type=float, default=0.0)
    parser.add_argument("--lesion-bank-size", type=int, default=32)
    parser.add_argument("--lesion-min-voxels", type=int, default=8)
    parser.add_argument("--lesion-edge-softening", type=int, default=3)
    parser.add_argument("--lesion-min-brain-coverage", type=float, default=0.0)
    parser.add_argument("--lesion-placement-attempts", type=int, default=8)
    parser.add_argument("--lesion-prefill-batches", type=int, default=0)
    parser.add_argument("--lesion-match-recipient-moments", action="store_true")
    parser.add_argument("--lesion-effect-margin", type=float, default=0.10)
    parser.add_argument("--lambda-lesion-paste-seg", type=float, default=0.0)
    parser.add_argument("--lambda-lesion-erase-seg", type=float, default=0.0)
    parser.add_argument("--lambda-lesion-mediator", type=float, default=0.0)
    parser.add_argument("--lambda-lesion-intervention-effect", type=float, default=0.0)
    parser.add_argument("--lambda-orthogonal", type=float, default=0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train causal MedNeXt on UTSW with disease/context proxy adjustment.")
    parser.add_argument("--config-json", help="Load saved trainer arguments before applying explicit CLI overrides.")
    parser.add_argument("--baseline-checkpoint", default="runs/mednext_utsw_s_k3/best.pt")
    parser.add_argument("--data-root", default="data/brats/PKG - UTSW-Glioma/UTSW-Glioma")
    parser.add_argument("--metadata-path")
    parser.add_argument("--splits-json")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--output-dir", default="runs/mednext_utsw_causal_s_k3")
    parser.add_argument("--model-id", choices=["S", "B", "M", "L"], default="S")
    parser.add_argument("--kernel-size", type=int, choices=[3, 5], default=3)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--volume-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--causal-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-context-bank-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--prefer-manual-seg", action="store_true")
    parser.add_argument("--use-ants-modalities", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--disk-cache-dir", help="Optional MedNeXt-only disk cache for preprocessed dataset items.")
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    _add_causal_args(parser)
    config_args, _ = parser.parse_known_args()
    if config_args.config_json:
        saved_config = _load_json(Path(config_args.config_json))
        destinations = {action.dest for action in parser._actions}
        parser.set_defaults(
            **{
                key: value
                for key, value in saved_config.items()
                if key in destinations and key != "config_json"
            }
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")
    splits = _load_or_make_splits(args, baseline_checkpoint, data_root)
    _save_json(splits, output_dir / "splits.json")
    _save_json(vars(args), output_dir / "config.json")
    _save_json(asdict(default_utsw_scm()), output_dir / "scm.json")

    train_loader = _make_loader(
        data_root,
        splits["train"],
        args,
        shuffle=True,
        cache_name="train",
        registered_modality_pair=_uses_registered_modality_training(args),
    )
    val_loader = _make_loader(
        data_root,
        splits["val"],
        args,
        shuffle=False,
        cache_name="val",
        registered_modality_pair=_uses_registered_modality_validation(args),
    )
    bank_loader = _make_loader(data_root, splits["train"], args, shuffle=False, cache_name="train")
    _require_metadata(train_loader.dataset, args.allow_missing_metadata)
    proxy_layout = _metadata_layout(train_loader.dataset)

    device = _resolve_device(args.device)
    model = _build_model_from_dataset(args, train_loader.dataset)
    _load_baseline_backbone(model, Path(args.baseline_checkpoint))
    init_report = _load_causal_init_checkpoint(model, args.init_checkpoint)
    if init_report:
        _save_json(init_report, output_dir / "init_checkpoint.json")
    model.to(device)
    optimizer = _build_optimizer(model, args)
    teacher_checkpoint = args.teacher_checkpoint
    if teacher_checkpoint is None and (
        args.lambda_teacher_distill > 0.0
        or args.lambda_adjusted_teacher_distill > 0.0
        or args.lambda_context_swap_teacher_distill > 0.0
        or args.lambda_region_causal_teacher_distill > 0.0
    ):
        teacher_checkpoint = args.baseline_checkpoint
    teacher_model = _build_teacher_model(teacher_checkpoint, device)

    best_monitor = float("-inf")
    best_monitor_key = "adjusted/brats/mean_dice"
    contrastive_bank: dict[str, Tensor] | None = None
    lesion_bank = (
        LesionInterventionBank(
            max_patches=args.lesion_bank_size,
            min_voxels=args.lesion_min_voxels,
            edge_softening=args.lesion_edge_softening,
            min_brain_coverage=args.lesion_min_brain_coverage,
            placement_attempts=args.lesion_placement_attempts,
            match_recipient_moments=args.lesion_match_recipient_moments,
        )
        if any(
            float(value) > 0.0
            for value in (
                args.lambda_lesion_paste_seg,
                args.lambda_lesion_erase_seg,
                args.lambda_lesion_intervention_effect,
            )
        )
        else None
    )
    lesion_prefill_report = prefill_lesion_intervention_bank(
        lesion_bank,
        bank_loader,
        max_batches=args.lesion_prefill_batches,
    )
    for epoch in range(1, args.epochs + 1):
        setattr(args, "_current_epoch", epoch)
        effective_et_veto_scale = _set_et_volume_veto_scale_for_epoch(model, args, epoch)
        registered_weight_alpha = _registered_modality_weight_schedule_alpha(args, epoch)
        _set_backbone_trainable(model, epoch > args.freeze_backbone_epochs)
        category_report = build_category_confounder_dictionary(
            model,
            bank_loader,
            device,
            max_batches=args.max_category_confounder_batches
            if args.max_category_confounder_batches is not None
            else args.max_context_bank_batches,
            threshold=args.threshold,
        )
        if contrastive_bank is None or (args.context_bank_refresh_epochs > 0 and (epoch - 1) % args.context_bank_refresh_epochs == 0):
            contrastive_bank = build_sdd_cite_bank(
                model,
                bank_loader,
                device,
                max_contexts=args.context_bank_size,
                max_batches=args.max_context_bank_batches,
                sampling=args.context_bank_sampling,
                seed=args.seed + epoch,
            )
        context_bank = None if contrastive_bank is None else contrastive_bank["z_c"]
        train_metrics = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            context_bank,
            contrastive_bank,
            proxy_layout,
            lesion_bank,
            teacher_model,
        )
        val_metrics = _run_eval_epoch(model, val_loader, device, args, context_bank, contrastive_bank, proxy_layout)
        train_metrics["schedule/et_volume_veto_scale"] = float(effective_et_veto_scale)
        val_metrics["schedule/et_volume_veto_scale"] = float(effective_et_veto_scale)
        train_metrics["schedule/registered_modality_weight_alpha"] = float(registered_weight_alpha)
        val_metrics["schedule/registered_modality_weight_alpha"] = float(registered_weight_alpha)
        monitor, monitor_key = checkpoint_monitor(val_metrics, args, prefer_adjusted=True)
        _save_json(
            {
                "epoch": epoch,
                "category_confounders": category_report,
                "lesion_bank": lesion_prefill_report,
                "train": train_metrics,
                "val": val_metrics,
            },
            output_dir / f"epoch_{epoch:03d}.json",
        )
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss/total"),
                "val_loss": val_metrics.get("loss/total"),
                "val_factual_mean_dice": val_metrics.get("brats/mean_dice"),
                "val_adjusted_mean_dice": val_metrics.get("adjusted/brats/mean_dice"),
                "val_registered_tta_mean_dice": val_metrics.get("registered_tta/brats/mean_dice"),
                "val_registered_consistency_stability": val_metrics.get("registered_consistency/stability_score"),
                "monitor": monitor,
                "monitor_key": monitor_key,
                "et_volume_veto_scale": effective_et_veto_scale,
            }
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "splits": splits,
            "proxy_dims": _metadata_dims(train_loader.dataset),
            "proxy_layout": proxy_layout,
            "category_confounders": category_report,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if monitor > best_monitor:
            best_monitor = monitor
            best_monitor_key = monitor_key
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_monitor": best_monitor, "best_val_monitor_key": best_monitor_key, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
