from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_cls: float = 1.0
    lambda_seg: float = 0.0
    lambda_region_supervision: float = 0.0
    lambda_seg_adjustment: float = 0.0
    lambda_region_adjustment: float = 0.0
    lambda_rec: float = 0.0
    lambda_dis: float = 0.0
    lambda_adjustment: float = 0.0
    lambda_cf_stability: float = 0.0
    lambda_seg_cf_stability: float = 0.0
    lambda_region_cf_stability: float = 0.0
    lambda_region_cf_contrastive: float = 0.0
    lambda_seg_disease_swap: float = 0.0
    lambda_region_disease_swap: float = 0.0
    lambda_disease_swap: float = 0.0
    context_stability_margin: float = 0.10
    adjustment_contexts: int | None = None
    contrastive_temperature: float = 0.20


class CounterfactualMemory:
    def __init__(
        self,
        num_samples: int,
        volume_ids: Sequence[int],
        latent_dim: int,
        device: torch.device,
        volume_momentum: float = 0.95,
        sample_momentum: float = 0.90,
        match_topk: int = 32,
        min_context_distance: float = 0.10,
        exemplar_capacity: int = 32,
        mixture_topk: int = 4,
        mixture_temperature: float = 0.35,
    ) -> None:
        unique_volumes = sorted({int(volume_id) for volume_id in volume_ids})
        if not unique_volumes:
            raise ValueError("CounterfactualMemory requires at least one volume id.")
        self.device = device
        self.sample_momentum = float(sample_momentum)
        self.volume_momentum = float(volume_momentum)
        self.match_topk = max(1, int(match_topk))
        self.min_context_distance = float(min_context_distance)
        self.exemplar_capacity = max(0, int(exemplar_capacity))
        self.mixture_topk = max(1, int(mixture_topk))
        self.mixture_temperature = max(float(mixture_temperature), 1e-4)
        self.volume_to_slot = {volume_id: index for index, volume_id in enumerate(unique_volumes)}
        self.sample_disease = torch.zeros(num_samples, latent_dim, device=device)
        self.sample_valid = torch.zeros(num_samples, dtype=torch.bool, device=device)
        self.sample_volume_slot = torch.full((num_samples,), -1, dtype=torch.long, device=device)
        self.volume_context = torch.zeros(len(unique_volumes), latent_dim, device=device)
        self.volume_valid = torch.zeros(len(unique_volumes), dtype=torch.bool, device=device)
        self._exemplar_device = torch.device("cpu")
        self.exemplar_ptr = 0
        self.exemplar_count = 0
        if self.exemplar_capacity > 0:
            self.exemplar_valid = torch.zeros(self.exemplar_capacity, dtype=torch.bool, device=self._exemplar_device)
            self.exemplar_sample_index = torch.full(
                (self.exemplar_capacity,),
                -1,
                dtype=torch.long,
                device=self._exemplar_device,
            )
            self.exemplar_volume_id = torch.full(
                (self.exemplar_capacity,),
                -1,
                dtype=torch.long,
                device=self._exemplar_device,
            )
            self.exemplar_z_d = torch.zeros(self.exemplar_capacity, latent_dim, device=self._exemplar_device)
            self.exemplar_z_c = torch.zeros(self.exemplar_capacity, latent_dim, device=self._exemplar_device)
            self.exemplar_has_label = torch.zeros(self.exemplar_capacity, dtype=torch.bool, device=self._exemplar_device)
            self.exemplar_has_mask = torch.zeros(self.exemplar_capacity, dtype=torch.bool, device=self._exemplar_device)
        else:
            self.exemplar_valid = torch.zeros(0, dtype=torch.bool, device=self._exemplar_device)
            self.exemplar_sample_index = torch.zeros(0, dtype=torch.long, device=self._exemplar_device)
            self.exemplar_volume_id = torch.zeros(0, dtype=torch.long, device=self._exemplar_device)
            self.exemplar_z_d = torch.zeros(0, latent_dim, device=self._exemplar_device)
            self.exemplar_z_c = torch.zeros(0, latent_dim, device=self._exemplar_device)
            self.exemplar_has_label = torch.zeros(0, dtype=torch.bool, device=self._exemplar_device)
            self.exemplar_has_mask = torch.zeros(0, dtype=torch.bool, device=self._exemplar_device)
        self.exemplar_labels: Tensor | None = None
        self.exemplar_masks: Tensor | None = None
        self.exemplar_disease_features: list[Tensor] | None = None

    def reset(self) -> None:
        self.sample_disease.zero_()
        self.sample_valid.zero_()
        self.sample_volume_slot.fill_(-1)
        self.volume_context.zero_()
        self.volume_valid.zero_()
        if self.exemplar_capacity > 0:
            self.exemplar_valid.zero_()
            self.exemplar_sample_index.fill_(-1)
            self.exemplar_volume_id.fill_(-1)
            self.exemplar_z_d.zero_()
            self.exemplar_z_c.zero_()
            self.exemplar_has_label.zero_()
            self.exemplar_has_mask.zero_()
        self.exemplar_ptr = 0
        self.exemplar_count = 0
        if self.exemplar_labels is not None:
            self.exemplar_labels.zero_()
        if self.exemplar_masks is not None:
            self.exemplar_masks.zero_()
        if self.exemplar_disease_features is not None:
            for feature in self.exemplar_disease_features:
                feature.zero_()

    def ready(self) -> bool:
        return bool(self.volume_valid.any() and self.sample_valid.any())

    def _volume_slots(self, volume_ids: Tensor) -> Tensor:
        slots = [self.volume_to_slot.get(int(volume_id), -1) for volume_id in volume_ids.detach().cpu().tolist()]
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def update(self, sample_indices: Tensor | None, volume_ids: Tensor | None, z_d: Tensor, z_c: Tensor) -> None:
        if sample_indices is None or volume_ids is None:
            return
        sample_indices = sample_indices.detach().long().to(self.device)
        volume_ids = volume_ids.detach().long().to(self.device)
        z_d = z_d.detach().to(self.device)
        z_c = z_c.detach().to(self.device)
        valid = (sample_indices >= 0) & (sample_indices < self.sample_disease.shape[0])
        if not torch.any(valid):
            return
        sample_indices = sample_indices[valid]
        volume_ids = volume_ids[valid]
        z_d = z_d[valid]
        z_c = z_c[valid]

        sample_slots = self._volume_slots(volume_ids)
        self.sample_volume_slot[sample_indices] = sample_slots
        seen_samples = self.sample_valid[sample_indices]
        updated_samples = z_d
        if torch.any(seen_samples):
            updated_samples = updated_samples.clone()
            seen_indices = torch.nonzero(seen_samples, as_tuple=False).flatten()
            updated_samples[seen_indices] = (
                self.sample_momentum * self.sample_disease[sample_indices[seen_indices]]
                + (1.0 - self.sample_momentum) * updated_samples[seen_indices]
            )
        self.sample_disease[sample_indices] = updated_samples
        self.sample_valid[sample_indices] = True

        unique_slots = torch.unique(sample_slots[sample_slots >= 0])
        for slot in unique_slots.tolist():
            slot_tensor = torch.tensor(slot, dtype=torch.long, device=self.device)
            slot_mask = sample_slots == slot_tensor
            context_mean = z_c[slot_mask].mean(dim=0)
            if bool(self.volume_valid[slot]):
                context_mean = self.volume_momentum * self.volume_context[slot] + (1.0 - self.volume_momentum) * context_mean
            self.volume_context[slot] = context_mean
            self.volume_valid[slot] = True

    def _ensure_exemplar_storage(
        self,
        mask: Tensor | None = None,
        label: Tensor | None = None,
        disease_features: Sequence[Tensor] | None = None,
    ) -> None:
        if self.exemplar_capacity <= 0:
            return
        if label is not None and self.exemplar_labels is None:
            self.exemplar_labels = torch.zeros(
                (self.exemplar_capacity, *label.shape[1:]),
                dtype=torch.float16,
                device=self._exemplar_device,
            )
        if mask is not None and self.exemplar_masks is None:
            self.exemplar_masks = torch.zeros(
                (self.exemplar_capacity, *mask.shape[1:]),
                dtype=torch.uint8,
                device=self._exemplar_device,
            )
        if disease_features is not None and self.exemplar_disease_features is None:
            self.exemplar_disease_features = [
                torch.zeros(
                    (self.exemplar_capacity, *feature.shape[1:]),
                    dtype=torch.float16,
                    device=self._exemplar_device,
                )
                for feature in disease_features
            ]

    def _find_exemplar_slot(self, sample_index: int) -> int | None:
        if self.exemplar_capacity <= 0:
            return None
        matches = torch.nonzero(
            self.exemplar_valid & (self.exemplar_sample_index == int(sample_index)),
            as_tuple=False,
        ).flatten()
        if matches.numel() == 0:
            return None
        return int(matches[0].item())

    def _allocate_exemplar_slot(self, sample_index: int) -> int:
        existing = self._find_exemplar_slot(sample_index)
        if existing is not None:
            return existing
        slot = self.exemplar_ptr
        self.exemplar_ptr = (self.exemplar_ptr + 1) % max(1, self.exemplar_capacity)
        self.exemplar_count = min(self.exemplar_count + 1, self.exemplar_capacity)
        return slot

    def store_exemplars(
        self,
        sample_indices: Tensor | None,
        volume_ids: Tensor | None,
        z_d: Tensor,
        z_c: Tensor,
        mask: Tensor | None = None,
        label: Tensor | None = None,
        disease_features: Sequence[Tensor] | None = None,
    ) -> None:
        if self.exemplar_capacity <= 0 or sample_indices is None or volume_ids is None:
            return
        sample_indices_cpu = sample_indices.detach().long().cpu()
        volume_ids_cpu = volume_ids.detach().long().cpu()
        z_d_cpu = z_d.detach().to(device=self._exemplar_device, dtype=torch.float32)
        z_c_cpu = z_c.detach().to(device=self._exemplar_device, dtype=torch.float32)
        mask_cpu = mask.detach().to(device=self._exemplar_device) if mask is not None else None
        label_cpu = label.detach().to(device=self._exemplar_device) if label is not None else None
        features_cpu = (
            [feature.detach().to(device=self._exemplar_device) for feature in disease_features]
            if disease_features is not None
            else None
        )
        self._ensure_exemplar_storage(mask_cpu, label_cpu, features_cpu)

        for row, sample_index in enumerate(sample_indices_cpu.tolist()):
            if sample_index < 0:
                continue
            slot = self._allocate_exemplar_slot(sample_index)
            self.exemplar_valid[slot] = True
            self.exemplar_sample_index[slot] = int(sample_index)
            self.exemplar_volume_id[slot] = int(volume_ids_cpu[row].item())
            self.exemplar_z_d[slot] = z_d_cpu[row]
            self.exemplar_z_c[slot] = z_c_cpu[row]
            if self.exemplar_labels is not None:
                if label_cpu is not None:
                    self.exemplar_labels[slot] = label_cpu[row].to(dtype=torch.float16)
                    self.exemplar_has_label[slot] = True
                else:
                    self.exemplar_has_label[slot] = False
            if self.exemplar_masks is not None:
                if mask_cpu is not None:
                    self.exemplar_masks[slot] = mask_cpu[row].to(dtype=torch.uint8)
                    self.exemplar_has_mask[slot] = True
                else:
                    self.exemplar_has_mask[slot] = False
            if self.exemplar_disease_features is not None:
                if features_cpu is not None:
                    for storage, feature in zip(self.exemplar_disease_features, features_cpu, strict=True):
                        storage[slot] = feature[row].to(dtype=torch.float16)
                else:
                    for storage in self.exemplar_disease_features:
                        storage[slot].zero_()

    def _valid_exemplar_slots(
        self,
        sample_indices: Tensor | None = None,
        require_mask: bool = False,
        require_label: bool = False,
    ) -> Tensor:
        if self.exemplar_capacity <= 0:
            return torch.zeros(0, dtype=torch.long, device=self._exemplar_device)
        valid = self.exemplar_valid.clone()
        if require_mask:
            valid = valid & self.exemplar_has_mask
        if require_label:
            valid = valid & self.exemplar_has_label
        if sample_indices is not None:
            blocked = sample_indices.detach().long().cpu()
            for sample_index in blocked.tolist():
                if sample_index >= 0:
                    valid = valid & (self.exemplar_sample_index != int(sample_index))
        return torch.nonzero(valid, as_tuple=False).flatten()

    def reference_latents(
        self,
        sample_indices: Tensor | None = None,
        device: torch.device | None = None,
        max_samples: int | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        valid_slots = self._valid_exemplar_slots(sample_indices=sample_indices)
        if valid_slots.numel() == 0:
            return None
        if max_samples is not None and max_samples > 0 and valid_slots.numel() > max_samples:
            positions = torch.linspace(
                0,
                valid_slots.numel() - 1,
                steps=max_samples,
                device=self._exemplar_device,
            )
            valid_slots = valid_slots[positions.round().long()]
        z_d = self.exemplar_z_d[valid_slots]
        z_c = self.exemplar_z_c[valid_slots]
        if device is not None:
            z_d = z_d.to(device)
            z_c = z_c.to(device)
        return z_d, z_c

    def lookup_volume_context(self, volume_ids: Tensor | None, fallback: Tensor) -> Tensor:
        if volume_ids is None:
            return fallback.detach()
        contexts = fallback.detach().to(self.device).clone()
        slots = self._volume_slots(volume_ids)
        valid_rows = torch.nonzero(slots >= 0, as_tuple=False).flatten()
        if valid_rows.numel() == 0:
            return contexts.to(fallback.device)
        valid_slots = slots[valid_rows]
        available = self.volume_valid[valid_slots]
        if torch.any(available):
            contexts[valid_rows[available]] = self.volume_context[valid_slots[available]]
        return contexts.to(fallback.device)

    def context_bank(self, max_contexts: int | None = None, device: torch.device | None = None) -> Tensor | None:
        valid_slots = torch.nonzero(self.volume_valid, as_tuple=False).flatten()
        if valid_slots.numel() == 0:
            return None
        if max_contexts is not None and max_contexts > 0 and valid_slots.numel() > max_contexts:
            positions = torch.linspace(0, valid_slots.numel() - 1, steps=max_contexts, device=self.device)
            valid_slots = valid_slots[positions.round().long()]
        bank = self.volume_context[valid_slots]
        if device is not None:
            bank = bank.to(device)
        return bank

    def matched_contexts(
        self,
        z_d: Tensor,
        current_context: Tensor,
        sample_indices: Tensor | None,
        volume_ids: Tensor | None,
    ) -> Tensor | None:
        if volume_ids is None:
            return None
        candidate_indices = torch.nonzero(self.sample_valid, as_tuple=False).flatten()
        if candidate_indices.numel() == 0 or not torch.any(self.volume_valid):
            return None
        candidate_disease = self.sample_disease[candidate_indices].to(z_d.device)
        candidate_slots = self.sample_volume_slot[candidate_indices]
        valid_context_mask = (candidate_slots >= 0) & self.volume_valid[candidate_slots]
        if not torch.any(valid_context_mask):
            return None
        candidate_indices = candidate_indices[valid_context_mask]
        candidate_disease = candidate_disease[valid_context_mask]
        candidate_slots = candidate_slots[valid_context_mask]
        candidate_context = self.volume_context[candidate_slots].to(z_d.device)
        candidate_sample_ids = candidate_indices.to(z_d.device)
        current_slots = self._volume_slots(volume_ids).to(z_d.device)
        current_sample_ids = sample_indices.detach().long().to(z_d.device) if sample_indices is not None else None
        current_context = current_context.detach()

        matches: list[Tensor] = []
        for row in range(z_d.shape[0]):
            mask = candidate_slots.to(z_d.device) != current_slots[row]
            if current_sample_ids is not None:
                mask = mask & (candidate_sample_ids != current_sample_ids[row])
            if not torch.any(mask):
                return None
            disease_dist = _cosine_distance(z_d[row], candidate_disease[mask])
            topk = min(self.match_topk, int(disease_dist.numel()))
            nearest = torch.topk(disease_dist, k=topk, largest=False).indices
            context_candidates = candidate_context[mask][nearest]
            context_dist = _cosine_distance(current_context[row], context_candidates)
            viable = context_dist >= self.min_context_distance
            if torch.any(viable):
                selected = torch.nonzero(viable, as_tuple=False).flatten()[torch.argmax(context_dist[viable])]
            else:
                selected = torch.argmax(context_dist)
            matches.append(context_candidates[selected])
        return torch.stack(matches, dim=0)

    def matched_disease_examples(
        self,
        z_d: Tensor,
        current_context: Tensor,
        sample_indices: Tensor | None,
        volume_ids: Tensor | None,
        require_mask: bool = False,
        require_label: bool = False,
        require_features: bool = False,
    ) -> dict[str, Tensor | tuple[Tensor, ...]] | None:
        valid_slots = self._valid_exemplar_slots(
            sample_indices=sample_indices,
            require_mask=require_mask,
            require_label=require_label,
        )
        if valid_slots.numel() == 0:
            return None
        if require_features and self.exemplar_disease_features is None:
            return None
        device = z_d.device
        exemplar_z_d = self.exemplar_z_d[valid_slots].to(device)
        exemplar_z_c = self.exemplar_z_c[valid_slots].to(device)
        exemplar_slots = valid_slots.to(device)
        exemplar_sample_ids = self.exemplar_sample_index[valid_slots].to(device)
        exemplar_volume_ids = self.exemplar_volume_id[valid_slots].to(device)
        current_sample_ids = sample_indices.detach().long().to(device) if sample_indices is not None else None
        current_volume_ids = volume_ids.detach().long().to(device) if volume_ids is not None else None

        def _weighted_mix(bank: Tensor, slots_cpu: Tensor, weights_cpu: Tensor, dtype: torch.dtype | None = None) -> Tensor:
            values = bank[slots_cpu]
            target_dtype = dtype if dtype is not None else values.dtype
            values = values.to(device=device, dtype=target_dtype)
            weights = weights_cpu.to(device=device, dtype=values.dtype)
            shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
            return (values * weights.view(shape)).sum(dim=0)

        primary_slots: list[int] = []
        mixture_slots: list[Tensor] = []
        mixture_weights: list[Tensor] = []
        for row in range(z_d.shape[0]):
            row_mask = torch.ones(exemplar_slots.shape[0], dtype=torch.bool, device=device)
            if current_sample_ids is not None:
                row_mask = row_mask & (exemplar_sample_ids != current_sample_ids[row])
            if current_volume_ids is not None:
                different_volume = row_mask & (exemplar_volume_ids != current_volume_ids[row])
                if torch.any(different_volume):
                    row_mask = different_volume
            if not torch.any(row_mask):
                return None
            context_dist = _cosine_distance(current_context[row], exemplar_z_c[row_mask])
            local_topk = min(self.match_topk, int(context_dist.numel()))
            nearest = torch.topk(context_dist, k=local_topk, largest=False).indices
            shortlisted_slots = exemplar_slots[row_mask][nearest]
            shortlisted_disease = exemplar_z_d[row_mask][nearest]
            shortlisted_context_dist = context_dist[nearest]
            disease_dist = _cosine_distance(z_d[row], shortlisted_disease)
            viable = disease_dist >= self.min_context_distance
            if torch.any(viable):
                shortlisted_slots = shortlisted_slots[viable]
                shortlisted_context_dist = shortlisted_context_dist[viable]
                disease_dist = disease_dist[viable]

            mix_topk = min(self.mixture_topk, int(shortlisted_slots.numel()))
            mixture_indices = torch.topk(disease_dist, k=mix_topk, largest=True).indices
            mixture_slot_tensor = shortlisted_slots[mixture_indices]
            mixture_scores = disease_dist[mixture_indices] - 0.25 * shortlisted_context_dist[mixture_indices]
            weights = torch.softmax(mixture_scores / self.mixture_temperature, dim=0)

            primary_slots.append(int(mixture_slot_tensor[torch.argmax(weights)].item()))
            mixture_slots.append(mixture_slot_tensor.to(self._exemplar_device))
            mixture_weights.append(weights.to(self._exemplar_device))

        slot_tensor_cpu = torch.tensor(primary_slots, dtype=torch.long, device=self._exemplar_device)
        examples: dict[str, Tensor | tuple[Tensor, ...]] = {
            "z_d": torch.stack(
                [_weighted_mix(self.exemplar_z_d, slots, weights, dtype=torch.float32) for slots, weights in zip(mixture_slots, mixture_weights)],
                dim=0,
            ),
            "z_c": torch.stack(
                [_weighted_mix(self.exemplar_z_c, slots, weights, dtype=torch.float32) for slots, weights in zip(mixture_slots, mixture_weights)],
                dim=0,
            ),
            "sample_index": self.exemplar_sample_index[slot_tensor_cpu].to(device),
            "volume_id": self.exemplar_volume_id[slot_tensor_cpu].to(device),
        }
        if require_label:
            if self.exemplar_labels is None:
                return None
            examples["label"] = torch.stack(
                [_weighted_mix(self.exemplar_labels, slots, weights, dtype=torch.float32) for slots, weights in zip(mixture_slots, mixture_weights)],
                dim=0,
            )
        if require_mask:
            if self.exemplar_masks is None:
                return None
            examples["mask"] = torch.stack(
                [_weighted_mix(self.exemplar_masks, slots, weights, dtype=torch.float32) for slots, weights in zip(mixture_slots, mixture_weights)],
                dim=0,
            )
        if require_features:
            if self.exemplar_disease_features is None:
                return None
            examples["disease_features"] = tuple(
                torch.stack(
                    [_weighted_mix(feature, slots, weights, dtype=torch.float32) for slots, weights in zip(mixture_slots, mixture_weights)],
                    dim=0,
                )
                for feature in self.exemplar_disease_features
            )
        return examples


def as_loss_weights(config: dict | LossWeights | None) -> LossWeights:
    if config is None:
        return LossWeights()
    if isinstance(config, LossWeights):
        return config
    fields = set(LossWeights.__dataclass_fields__)
    return LossWeights(**{key: value for key, value in config.items() if key in fields})


def classification_loss(logits: Tensor, target: Tensor) -> Tensor:
    if target.ndim == 1:
        target = target.unsqueeze(1)
    return F.binary_cross_entropy_with_logits(logits, target.float())


def dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(2, logits.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    return bce + dice_loss(logits, target)


def _derived_brats_regions(probs: Tensor) -> Tensor:
    if probs.shape[1] < 3:
        raise ValueError(f"BraTS region supervision expects three channels, got shape {tuple(probs.shape)}")
    wt = 1.0 - torch.prod(1.0 - probs[:, [0, 1, 2]], dim=1, keepdim=True)
    tc = 1.0 - torch.prod(1.0 - probs[:, [0, 2]], dim=1, keepdim=True)
    et = probs[:, [2]]
    return torch.cat([wt, tc, et], dim=1)


def brats_region_supervision_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    probs = torch.sigmoid(logits)
    target = target.float()
    region_probs = _derived_brats_regions(probs)
    region_targets = _derived_brats_regions(target)
    bce = F.binary_cross_entropy(region_probs, region_targets)
    dims = tuple(range(2, logits.ndim))
    intersection = (region_probs * region_targets).sum(dim=dims)
    denominator = region_probs.sum(dim=dims) + region_targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return bce + (1.0 - dice.mean())


def brats_region_stability_loss(original_logits: Tensor, counterfactual_logits: Tensor, margin: float) -> Tensor:
    original_regions = _derived_brats_regions(torch.sigmoid(original_logits))
    counterfactual_regions = _derived_brats_regions(torch.sigmoid(counterfactual_logits))
    return bounded_stability_loss(original_regions, counterfactual_regions, margin)


def reconstruction_loss(reconstruction: Tensor, image: Tensor) -> Tensor:
    return F.l1_loss(reconstruction, image)


def decorrelation_loss(
    z_d: Tensor,
    z_c: Tensor,
    reference_latents: tuple[Tensor, Tensor] | None = None,
    eps: float = 1e-6,
) -> Tensor:
    if reference_latents is not None:
        ref_z_d, ref_z_c = reference_latents
        if ref_z_d.numel() > 0 and ref_z_c.numel() > 0:
            z_d = torch.cat([z_d, ref_z_d.to(device=z_d.device, dtype=z_d.dtype)], dim=0)
            z_c = torch.cat([z_c, ref_z_c.to(device=z_c.device, dtype=z_c.dtype)], dim=0)
    if z_d.shape[0] < 2:
        return z_d.new_zeros(())
    z_d = z_d - z_d.mean(dim=0, keepdim=True)
    z_c = z_c - z_c.mean(dim=0, keepdim=True)
    z_d = z_d / (z_d.std(dim=0, keepdim=True) + eps)
    z_c = z_c / (z_c.std(dim=0, keepdim=True) + eps)
    corr = z_d.T @ z_c / (z_d.shape[0] - 1)
    return corr.square().mean()


def shifted_permutation(batch_size: int, device: torch.device) -> Tensor:
    if batch_size <= 1:
        return torch.arange(batch_size, device=device)
    return torch.roll(torch.arange(batch_size, device=device), shifts=1)


def bounded_stability_loss(original: Tensor, counterfactual: Tensor, margin: float) -> Tensor:
    delta = (original - counterfactual).abs()
    return torch.relu(delta - margin).mean()


def _cosine_distance(query: Tensor, bank: Tensor, eps: float = 1e-6) -> Tensor:
    if bank.ndim == 1:
        bank = bank.unsqueeze(0)
    query = F.normalize(query.unsqueeze(0), dim=-1, eps=eps)
    bank = F.normalize(bank, dim=-1, eps=eps)
    return 1.0 - torch.sum(query * bank, dim=-1)


def _pairwise_cosine_distance(latents: Tensor, eps: float = 1e-6) -> Tensor:
    normalized = F.normalize(latents, dim=-1, eps=eps)
    return 1.0 - normalized @ normalized.T


def _matched_batch_permutation(
    primary_latents: Tensor,
    secondary_latents: Tensor,
    volume_ids: Tensor | None = None,
    topk: int = 3,
) -> Tensor:
    batch_size = int(primary_latents.shape[0])
    device = primary_latents.device
    if batch_size <= 1:
        return torch.arange(batch_size, device=device)

    primary_distance = _pairwise_cosine_distance(primary_latents)
    secondary_distance = _pairwise_cosine_distance(secondary_latents)
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    primary_distance = primary_distance.masked_fill(eye, float("inf"))
    permutation = torch.arange(batch_size, device=device)

    for row in range(batch_size):
        candidate_mask = ~eye[row]
        if volume_ids is not None:
            different_volume = candidate_mask & (volume_ids != volume_ids[row])
            if torch.any(different_volume):
                candidate_mask = different_volume
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        candidate_primary = primary_distance[row, candidate_indices]
        candidate_secondary = secondary_distance[row, candidate_indices]
        local_topk = min(max(1, topk), int(candidate_indices.numel()))
        nearest = torch.topk(candidate_primary, k=local_topk, largest=False).indices
        shortlisted = candidate_indices[nearest]
        best = shortlisted[torch.argmax(candidate_secondary[nearest])]
        permutation[row] = best
    return permutation


def matched_context_permutation(z_d: Tensor, z_c: Tensor, volume_ids: Tensor | None = None) -> Tensor:
    return _matched_batch_permutation(z_d, z_c, volume_ids=volume_ids, topk=3)


def matched_disease_permutation(z_d: Tensor, z_c: Tensor, volume_ids: Tensor | None = None) -> Tensor:
    return _matched_batch_permutation(z_c, z_d, volume_ids=volume_ids, topk=3)


def _context_subset(z_c: Tensor, max_contexts: int | None) -> Tensor:
    if max_contexts is None or max_contexts <= 0 or max_contexts >= z_c.shape[0]:
        return z_c
    return z_c[:max_contexts]


def _repeat_feature_bank(features: tuple[Tensor, ...] | list[Tensor], repeats: int) -> tuple[Tensor, ...]:
    repeated: list[Tensor] = []
    for feature in features:
        expanded = feature[:, None, ...].expand(feature.shape[0], repeats, *feature.shape[1:])
        repeated.append(expanded.reshape(feature.shape[0] * repeats, *feature.shape[1:]))
    return tuple(repeated)


def _resolve_context_bank(z_c: Tensor, context_bank: Tensor | None, max_contexts: int | None) -> Tensor:
    bank = z_c if context_bank is None else context_bank.to(z_c.device)
    return _context_subset(bank, max_contexts)


def backdoor_adjusted_logits(
    model: nn.Module,
    z_d: Tensor,
    z_c: Tensor,
    max_contexts: int | None = None,
    context_bank: Tensor | None = None,
) -> Tensor:
    """Approximate p(y | do(z_d)) by averaging over a context bank from the batch."""

    context_bank = _resolve_context_bank(z_c, context_bank, max_contexts)
    batch_size, num_contexts = z_d.shape[0], context_bank.shape[0]
    z_d_rep = z_d[:, None, :].expand(batch_size, num_contexts, -1).reshape(batch_size * num_contexts, -1)
    z_c_rep = context_bank[None, :, :].expand(batch_size, num_contexts, -1).reshape(batch_size * num_contexts, -1)
    logits = model.predict_from_latents(z_d_rep, z_c_rep)["logits"]
    probs = torch.sigmoid(logits).view(batch_size, num_contexts, -1).mean(dim=1)
    probs = probs.clamp(min=1e-5, max=1.0 - 1e-5)
    return torch.logit(probs)


def backdoor_adjusted_seg_logits(
    model: nn.Module,
    disease_features: tuple[Tensor, ...] | list[Tensor] | None,
    z_d: Tensor,
    z_c: Tensor,
    max_contexts: int | None = None,
    context_bank: Tensor | None = None,
) -> Tensor:
    """Approximate p(s | do(z_d)) by averaging segmentation probabilities over sampled contexts."""

    context_bank = _resolve_context_bank(z_c, context_bank, max_contexts)
    batch_size, num_contexts = z_d.shape[0], context_bank.shape[0]
    z_d_rep = z_d[:, None, :].expand(batch_size, num_contexts, -1).reshape(batch_size * num_contexts, -1)
    z_c_rep = context_bank[None, :, :].expand(batch_size, num_contexts, -1).reshape(batch_size * num_contexts, -1)
    repeated_features = _repeat_feature_bank(disease_features, num_contexts) if disease_features is not None else None
    seg_logits = model.segment_from_parts(z_d_rep, z_c_rep, repeated_features)
    probs = torch.sigmoid(seg_logits).view(batch_size, num_contexts, *seg_logits.shape[1:]).mean(dim=1)
    probs = probs.clamp(min=1e-5, max=1.0 - 1e-5)
    return torch.logit(probs)


def _region_attention_maps(weights: Tensor) -> Tensor:
    if weights.ndim == 3:
        weights = weights.unsqueeze(1)
    weights = weights.float().clamp(min=0.0, max=1.0)
    if weights.shape[1] == 3:
        return _derived_brats_regions(weights)
    return weights


def lesion_positive_slices(weights: Tensor) -> Tensor:
    region_weights = _region_attention_maps(weights)
    return region_weights.flatten(start_dim=1).sum(dim=1) > 0.0


def lesion_aware_region_descriptor(seg_features: Tensor, lesion_weights: Tensor, eps: float = 1e-6) -> Tensor:
    region_weights = _region_attention_maps(lesion_weights)
    if region_weights.shape[-2:] != seg_features.shape[-2:]:
        region_weights = F.interpolate(
            region_weights,
            size=seg_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    pooled_features: list[Tensor] = [seg_features.mean(dim=(2, 3))]
    for region_index in range(region_weights.shape[1]):
        weights = region_weights[:, region_index : region_index + 1]
        norm = weights.sum(dim=(2, 3)).clamp_min(eps)
        pooled = (seg_features * weights).sum(dim=(2, 3)) / norm
        pooled_features.append(pooled)
    descriptor = torch.cat(pooled_features, dim=1)
    return F.normalize(descriptor, dim=1, eps=eps)


def counterfactual_contrastive_loss(
    anchor: Tensor,
    positive: Tensor,
    hard_negative: Tensor,
    temperature: float = 0.20,
) -> Tensor:
    if anchor.shape[0] == 0:
        return anchor.new_zeros(())
    temperature = max(float(temperature), 1e-4)
    anchor = F.normalize(anchor, dim=1, eps=1e-6)
    positive = F.normalize(positive, dim=1, eps=1e-6)
    hard_negative = F.normalize(hard_negative, dim=1, eps=1e-6)
    logits = torch.cat(
        [
            torch.sum(anchor * positive, dim=1, keepdim=True),
            torch.sum(anchor * hard_negative, dim=1, keepdim=True),
        ],
        dim=1,
    ) / temperature
    targets = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, targets)


def backdoor_adjusted_seg_descriptors(
    model: nn.Module,
    disease_features: tuple[Tensor, ...] | list[Tensor] | None,
    z_d: Tensor,
    z_c: Tensor,
    lesion_weights: Tensor,
    max_contexts: int | None = None,
    context_bank: Tensor | None = None,
) -> Tensor:
    context_bank = _resolve_context_bank(z_c, context_bank, max_contexts)
    batch_size, num_contexts = z_d.shape[0], context_bank.shape[0]
    z_d_rep = z_d[:, None, :].expand(batch_size, num_contexts, -1).reshape(batch_size * num_contexts, -1)
    z_c_rep = context_bank[None, :, :].expand(batch_size, num_contexts, -1).reshape(batch_size * num_contexts, -1)
    repeated_features = _repeat_feature_bank(disease_features, num_contexts) if disease_features is not None else None
    repeated_weights = (
        lesion_weights[:, None, ...]
        .expand(batch_size, num_contexts, *lesion_weights.shape[1:])
        .reshape(batch_size * num_contexts, *lesion_weights.shape[1:])
    )
    seg_outputs = model.segment_outputs_from_parts(z_d_rep, z_c_rep, repeated_features)
    descriptors = lesion_aware_region_descriptor(seg_outputs["seg_features"], repeated_weights)
    return descriptors.view(batch_size, num_contexts, -1).mean(dim=1)


def compute_crn_losses(
    model: nn.Module,
    batch: dict[str, Tensor],
    outputs: dict[str, Tensor],
    weights: dict | LossWeights | None = None,
    counterfactual_memory: CounterfactualMemory | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    cfg = as_loss_weights(weights)
    terms: dict[str, Tensor] = {}
    zero = outputs["z_d"].new_zeros(())
    can_counterfactual = outputs["z_d"].shape[0] > 1
    sample_indices = batch.get("index")
    volume_ids = batch.get("volume")
    memory_context_bank = (
        counterfactual_memory.context_bank(cfg.adjustment_contexts, device=outputs["z_d"].device)
        if counterfactual_memory is not None
        else None
    )
    memory_reference_latents = (
        counterfactual_memory.reference_latents(
            sample_indices=sample_indices,
            device=outputs["z_d"].device,
            max_samples=cfg.adjustment_contexts,
        )
        if counterfactual_memory is not None
        else None
    )
    adjusted_seg_logits: Tensor | None = None
    cf_seg_logits: Tensor | None = None
    donor_seg_logits: Tensor | None = None
    donor_seg_features: Tensor | None = None
    matched_contexts: Tensor | None = None
    donor_examples: dict[str, Tensor | tuple[Tensor, ...]] | None = None

    if cfg.lambda_cls and "label" in batch and "logits" in outputs:
        terms["cls"] = classification_loss(outputs["logits"], batch["label"])
    else:
        terms["cls"] = zero

    if cfg.lambda_seg and "mask" in batch and "seg_logits" in outputs:
        terms["seg"] = segmentation_loss(outputs["seg_logits"], batch["mask"])
    else:
        terms["seg"] = zero

    if cfg.lambda_region_supervision and "mask" in batch and "seg_logits" in outputs and batch["mask"].shape[1] == 3:
        terms["region_supervision"] = brats_region_supervision_loss(outputs["seg_logits"], batch["mask"])
    else:
        terms["region_supervision"] = zero

    needs_seg_adjustment = (
        (cfg.lambda_seg_adjustment or cfg.lambda_region_adjustment)
        and "mask" in batch
        and "seg_logits" in outputs
    )
    if needs_seg_adjustment:
        adjusted_seg_logits = backdoor_adjusted_seg_logits(
            model,
            outputs.get("disease_features"),
            outputs["z_d"],
            outputs["z_c"],
            cfg.adjustment_contexts,
            context_bank=memory_context_bank,
        )

    if cfg.lambda_seg_adjustment and adjusted_seg_logits is not None:
        terms["seg_adjustment"] = segmentation_loss(adjusted_seg_logits, batch["mask"])
    else:
        terms["seg_adjustment"] = zero

    if cfg.lambda_region_adjustment and adjusted_seg_logits is not None and batch["mask"].shape[1] == 3:
        terms["region_adjustment"] = brats_region_supervision_loss(adjusted_seg_logits, batch["mask"])
    else:
        terms["region_adjustment"] = zero

    if cfg.lambda_rec and "reconstruction" in outputs:
        image = batch["image"]
        if image.ndim == 5 and outputs["reconstruction"].ndim == 4:
            center_idx = image.shape[2] // 2
            image = image[:, :, center_idx]
        terms["rec"] = reconstruction_loss(outputs["reconstruction"], image)
    else:
        terms["rec"] = zero

    if cfg.lambda_dis:
        terms["dis"] = decorrelation_loss(outputs["z_d"], outputs["z_c"], reference_latents=memory_reference_latents)
    else:
        terms["dis"] = zero

    if cfg.lambda_adjustment and "label" in batch and "logits" in outputs:
        adjusted = backdoor_adjusted_logits(
            model,
            outputs["z_d"],
            outputs["z_c"],
            cfg.adjustment_contexts,
            context_bank=memory_context_bank,
        )
        terms["adjustment"] = classification_loss(adjusted, batch["label"])
    else:
        terms["adjustment"] = zero

    if cfg.lambda_cf_stability or cfg.lambda_seg_cf_stability or cfg.lambda_region_cf_stability:
        if counterfactual_memory is not None:
            matched_contexts = counterfactual_memory.matched_contexts(outputs["z_d"], outputs["z_c"], sample_indices, volume_ids)
        if matched_contexts is None and can_counterfactual:
            perm = matched_context_permutation(outputs["z_d"], outputs["z_c"], volume_ids)
            matched_contexts = outputs["z_c"][perm]

    if cfg.lambda_cf_stability and matched_contexts is not None and "logits" in outputs:
        cf_logits = model.predict_from_latents(outputs["z_d"], matched_contexts)["logits"]
        terms["cf_stability"] = bounded_stability_loss(
            torch.sigmoid(outputs["logits"]),
            torch.sigmoid(cf_logits),
            cfg.context_stability_margin,
        )
    else:
        terms["cf_stability"] = zero

    needs_seg_cf = (cfg.lambda_seg_cf_stability or cfg.lambda_region_cf_stability) and matched_contexts is not None and "seg_logits" in outputs
    if needs_seg_cf:
        cf_disease_features = outputs.get("disease_features")
        cf_seg_logits = model.segment_from_parts(outputs["z_d"], matched_contexts, cf_disease_features)

    if cfg.lambda_seg_cf_stability and cf_seg_logits is not None:
        terms["seg_cf_stability"] = bounded_stability_loss(
            torch.sigmoid(outputs["seg_logits"]),
            torch.sigmoid(cf_seg_logits),
            cfg.context_stability_margin,
        )
    else:
        terms["seg_cf_stability"] = zero

    if cfg.lambda_region_cf_stability and cf_seg_logits is not None and "mask" in batch and batch["mask"].shape[1] == 3:
        terms["region_cf_stability"] = brats_region_stability_loss(
            outputs["seg_logits"],
            cf_seg_logits,
            cfg.context_stability_margin,
        )
    else:
        terms["region_cf_stability"] = zero

    if cfg.lambda_disease_swap and "label" in batch and "logits" in outputs:
        if counterfactual_memory is not None:
            donor_examples = counterfactual_memory.matched_disease_examples(
                outputs["z_d"],
                outputs["z_c"],
                sample_indices,
                volume_ids,
                require_label=True,
            )
        if donor_examples is not None:
            donor_logits = model.predict_from_latents(donor_examples["z_d"], outputs["z_c"])["logits"]
            terms["disease_swap"] = classification_loss(donor_logits, donor_examples["label"])
        elif can_counterfactual:
            perm = matched_disease_permutation(outputs["z_d"], outputs["z_c"], volume_ids)
            donor_logits = model.predict_from_latents(outputs["z_d"][perm], outputs["z_c"])["logits"]
            terms["disease_swap"] = classification_loss(donor_logits, batch["label"][perm])
        else:
            terms["disease_swap"] = zero
    else:
        terms["disease_swap"] = zero

    needs_disease_swap = (
        (cfg.lambda_seg_disease_swap or cfg.lambda_region_disease_swap or cfg.lambda_region_cf_contrastive)
        and "mask" in batch
        and "seg_logits" in outputs
    )
    donor_mask: Tensor | None = None
    if needs_disease_swap:
        requires_features = not getattr(model, "supports_latent_segmentation", False)
        if counterfactual_memory is not None:
            donor_examples = counterfactual_memory.matched_disease_examples(
                outputs["z_d"],
                outputs["z_c"],
                sample_indices,
                volume_ids,
                require_mask=True,
                require_features=requires_features,
            )
        if donor_examples is not None:
            donor_features = donor_examples.get("disease_features")
            donor_outputs = model.segment_outputs_from_parts(donor_examples["z_d"], outputs["z_c"], donor_features)
            donor_seg_logits = donor_outputs["seg_logits"]
            donor_seg_features = donor_outputs.get("seg_features")
            donor_mask = donor_examples["mask"]
        elif can_counterfactual:
            perm = matched_disease_permutation(outputs["z_d"], outputs["z_c"], volume_ids)
            donor_features = outputs.get("disease_features")
            if donor_features is not None:
                donor_features = tuple(feature[perm] for feature in donor_features)
            donor_outputs = model.segment_outputs_from_parts(outputs["z_d"][perm], outputs["z_c"], donor_features)
            donor_seg_logits = donor_outputs["seg_logits"]
            donor_seg_features = donor_outputs.get("seg_features")
            donor_mask = batch["mask"][perm]

    if cfg.lambda_seg_disease_swap and donor_seg_logits is not None and donor_mask is not None:
        terms["seg_disease_swap"] = segmentation_loss(donor_seg_logits, donor_mask)
    else:
        terms["seg_disease_swap"] = zero

    if cfg.lambda_region_disease_swap and donor_seg_logits is not None and donor_mask is not None and batch["mask"].shape[1] == 3:
        terms["region_disease_swap"] = brats_region_supervision_loss(donor_seg_logits, donor_mask)
    else:
        terms["region_disease_swap"] = zero

    if (
        cfg.lambda_region_cf_contrastive
        and "mask" in batch
        and batch["mask"].shape[1] == 3
        and "seg_features" in outputs
        and donor_mask is not None
        and donor_seg_features is not None
    ):
        active_rows = lesion_positive_slices(batch["mask"]) & lesion_positive_slices(donor_mask)
        if torch.any(active_rows):
            anchor_descriptor = lesion_aware_region_descriptor(
                outputs["seg_features"][active_rows],
                batch["mask"][active_rows],
            )
            positive_descriptor = backdoor_adjusted_seg_descriptors(
                model,
                tuple(feature[active_rows] for feature in outputs.get("disease_features", ())),
                outputs["z_d"][active_rows],
                outputs["z_c"][active_rows],
                batch["mask"][active_rows],
                max_contexts=cfg.adjustment_contexts,
                context_bank=memory_context_bank,
            )
            negative_descriptor = lesion_aware_region_descriptor(
                donor_seg_features[active_rows],
                donor_mask[active_rows],
            )
            terms["region_cf_contrastive"] = counterfactual_contrastive_loss(
                anchor_descriptor,
                positive_descriptor,
                negative_descriptor,
                temperature=cfg.contrastive_temperature,
            )
        else:
            terms["region_cf_contrastive"] = zero
    else:
        terms["region_cf_contrastive"] = zero

    weighted_terms = {
        "cls": cfg.lambda_cls * terms["cls"],
        "seg": cfg.lambda_seg * terms["seg"],
        "region_supervision": cfg.lambda_region_supervision * terms["region_supervision"],
        "seg_adjustment": cfg.lambda_seg_adjustment * terms["seg_adjustment"],
        "region_adjustment": cfg.lambda_region_adjustment * terms["region_adjustment"],
        "rec": cfg.lambda_rec * terms["rec"],
        "dis": cfg.lambda_dis * terms["dis"],
        "adjustment": cfg.lambda_adjustment * terms["adjustment"],
        "cf_stability": cfg.lambda_cf_stability * terms["cf_stability"],
        "seg_cf_stability": cfg.lambda_seg_cf_stability * terms["seg_cf_stability"],
        "region_cf_stability": cfg.lambda_region_cf_stability * terms["region_cf_stability"],
        "region_cf_contrastive": cfg.lambda_region_cf_contrastive * terms["region_cf_contrastive"],
        "seg_disease_swap": cfg.lambda_seg_disease_swap * terms["seg_disease_swap"],
        "region_disease_swap": cfg.lambda_region_disease_swap * terms["region_disease_swap"],
        "disease_swap": cfg.lambda_disease_swap * terms["disease_swap"],
    }
    total = sum(weighted_terms.values(), zero)
    logs = {f"loss/{name}": value.detach() for name, value in terms.items()}
    logs.update({f"weighted/{name}": value.detach() for name, value in weighted_terms.items()})
    logs["loss/total"] = total.detach()
    return total, logs
