from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mednext.causal import CausalMedNeXt, build_causal_mednext
from baselines.mednext.common import load_model_for_eval, main_logits
from baselines.segformer3d.causal import default_utsw_scm
from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_causal_utsw import (
    _add_context_swap_outputs,
    _build_optimizer,
    _causal_loss_terms,
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
    _segmentation_terms,
    _spatial_disease_attention_loss,
    _spatial_region_head_loss,
    _subsample_context_bank,
    _weighted_total,
)
from baselines.segformer3d.train_utsw import _average_metric_dicts, _case_ids, _make_splits, _resolve_device, _save_json
from crn.metrics import brats_region_metrics


def _make_loader(root: Path, case_ids: list[str], args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = UTSWGliomaDataset(
        root=root,
        volume_size=args.volume_size,
        case_ids=case_ids,
        crop_margin=args.crop_margin,
        prefer_manual_seg=args.prefer_manual_seg,
        use_ants_modalities=args.use_ants_modalities,
        metadata_path=args.metadata_path,
        include_metadata=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
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
        treatment_proxy_dim=treatment_proxy_dim,
        **_metadata_dims(dataset),
    )


def _load_baseline_backbone(model: CausalMedNeXt, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_baseline_state_dict(state_dict, strict_backbone=True)
    return checkpoint


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
            adversary_strength=args.adversary_strength,
            max_contrastive_negatives=args.cite_bank_negatives,
        )
        model.add_cite_outputs(outputs, contrastive_bank, max_negatives=args.cite_bank_negatives)
        _add_context_swap_outputs(model, outputs, bank_device, args)
        teacher_logits = _teacher_logits(teacher_model, image)
        if teacher_logits is not None:
            outputs["teacher_logits"] = teacher_logits
        terms = _causal_loss_terms(outputs, batch, args, proxy_layout)
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
    bank_device = context_bank.to(device) if context_bank is not None else None
    for batch_idx, batch in enumerate(tqdm(loader, desc="mednext-causal-val", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        outputs = model(
            image,
            context_bank=bank_device,
            max_adjustment_contexts=args.adjustment_contexts,
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
        metrics = brats_region_metrics(outputs["logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold)
        if "adjusted_logits" in outputs:
            metrics.update(_prefix_metrics(brats_region_metrics(outputs["adjusted_logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold), "adjusted"))
        if "context_swap_logits" in outputs:
            metrics.update(_prefix_metrics(brats_region_metrics(outputs["context_swap_logits"].detach().cpu(), target.detach().cpu(), threshold=args.threshold), "context_swap"))
        metric_items.append(metrics)
        if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
            break
    return {**_average_metric_dicts(loss_logs), **_average_metric_dicts(metric_items)}


def _add_causal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context-bank-size", type=int, default=64)
    parser.add_argument("--context-bank-sampling", choices=["uniform", "random", "farthest"], default="uniform")
    parser.add_argument("--adjustment-contexts", type=int, default=4)
    parser.add_argument("--context-swap-strategy", choices=["none", "random", "nearest", "farthest"], default="none")
    parser.add_argument("--context-bank-refresh-epochs", type=int, default=1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
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
    parser.add_argument("--channel-loss-weights", default="1.0,1.0,1.0")
    parser.add_argument("--region-loss-weights", default="1.0,1.5,2.5")
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
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    _add_causal_args(parser)
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

    train_loader = _make_loader(data_root, splits["train"], args, shuffle=True)
    val_loader = _make_loader(data_root, splits["val"], args, shuffle=False)
    bank_loader = _make_loader(data_root, splits["train"], args, shuffle=False)
    _require_metadata(train_loader.dataset, args.allow_missing_metadata)
    proxy_layout = _metadata_layout(train_loader.dataset)

    device = _resolve_device(args.device)
    model = _build_model_from_dataset(args, train_loader.dataset)
    _load_baseline_backbone(model, Path(args.baseline_checkpoint))
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

    best_adjusted_dice = float("-inf")
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
        monitor = float(val_metrics.get("adjusted/brats/mean_dice", val_metrics.get("brats/mean_dice", float("-inf"))))
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
        if monitor > best_adjusted_dice:
            best_adjusted_dice = monitor
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_adjusted_brats_mean_dice": best_adjusted_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
