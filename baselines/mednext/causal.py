from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from baselines.mednext.model import MedNeXtSegmenter, build_mednext_segmenter
from baselines.segformer3d.causal.model import LatentFeatureModulator


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * grad_output, None


def gradient_reverse(value: Tensor, strength: float = 1.0) -> Tensor:
    return _GradientReverse.apply(value, float(strength))


class CausalMedNeXt(nn.Module):
    """MedNeXt backbone with disease/context latent SCM-style adjustment.

    The output contract intentionally matches `CausalSegFormer3D`: factual
    logits, disease/context latents, optional proxy predictions, and optional
    context-bank-adjusted logits.
    """

    def __init__(
        self,
        model_id: str = "S",
        kernel_size: int = 3,
        in_channels: int = 4,
        num_classes: int = 3,
        latent_dim: int = 128,
        context_proxy_dim: int = 0,
        disease_proxy_dim: int = 0,
        annotation_proxy_dim: int = 0,
        treatment_proxy_dim: int = 2,
        base_channels: int | None = None,
        modulation_scale: float = 0.1,
        causal_residual_scale: float = 0.2,
        contrastive_dim: int = 64,
        spatial_refiner_scale: float = 0.5,
        region_fusion_scale: float = 0.0,
        prototype_dim: int = 32,
        prototype_fusion_scale: float = 0.0,
        prototype_temperature: float = 0.1,
        category_confounder_scale: float = 0.0,
        category_confounder_temperature: float = 0.2,
        modality_prior_scale: float = 0.0,
        logit_calibration_scale: float = 0.0,
        cascade_refiner_scale: float = 0.0,
        frontdoor_mediator_scale: float = 0.0,
        frontdoor_residual_scale: float = 0.25,
        use_causal_mediator_router: bool = False,
        use_nested_causal_intervention: bool = False,
        nested_causal_gate_scale: float = 1.0,
        region_causal_bottleneck_scale: float = 0.0,
        region_causal_background_leak: float = 0.05,
        region_causal_base: str = "prior",
        region_causal_mask_source: str = "spatial",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.contrastive_dim = int(contrastive_dim)
        self.num_classes = int(num_classes)
        self.prototype_dim = max(4, int(prototype_dim))
        self.context_proxy_dim = int(context_proxy_dim)
        self.disease_proxy_dim = int(disease_proxy_dim)
        self.annotation_proxy_dim = int(annotation_proxy_dim)
        self.treatment_proxy_dim = int(treatment_proxy_dim)
        self.causal_residual_scale = float(causal_residual_scale)
        self.spatial_refiner_scale = float(spatial_refiner_scale)
        self.region_fusion_scale = float(region_fusion_scale)
        self.prototype_fusion_scale = float(prototype_fusion_scale)
        self.prototype_temperature = max(float(prototype_temperature), 1e-6)
        self.category_confounder_scale = max(0.0, float(category_confounder_scale))
        self.category_confounder_temperature = max(float(category_confounder_temperature), 1e-6)
        self.modality_prior_scale = max(0.0, float(modality_prior_scale))
        self.logit_calibration_scale = max(0.0, float(logit_calibration_scale))
        self.cascade_refiner_scale = max(0.0, float(cascade_refiner_scale))
        self.frontdoor_mediator_scale = min(max(0.0, float(frontdoor_mediator_scale)), 1.0)
        self.frontdoor_residual_scale = max(0.0, float(frontdoor_residual_scale))
        self.use_causal_mediator_router = bool(use_causal_mediator_router)
        self.use_nested_causal_intervention = bool(use_nested_causal_intervention)
        self.nested_causal_gate_scale = min(max(0.0, float(nested_causal_gate_scale)), 1.0)
        self.region_causal_bottleneck_scale = min(max(0.0, float(region_causal_bottleneck_scale)), 1.0)
        self.region_causal_background_leak = min(max(0.0, float(region_causal_background_leak)), 1.0)
        if region_causal_base not in {"prior", "factual"}:
            raise ValueError(f"region_causal_base must be 'prior' or 'factual', got {region_causal_base!r}")
        self.region_causal_base = str(region_causal_base)
        if region_causal_mask_source not in {"spatial", "factual"}:
            raise ValueError(
                "region_causal_mask_source must be 'spatial' or 'factual', "
                f"got {region_causal_mask_source!r}"
            )
        self.region_causal_mask_source = str(region_causal_mask_source)
        self.backbone = build_mednext_segmenter(
            model_id=model_id,
            kernel_size=kernel_size,
            in_channels=in_channels,
            num_classes=num_classes,
            deep_supervision=False,
            base_channels=base_channels,
        )
        self.feature_channels = tuple(int(channel) for channel in self.backbone.feature_channels)
        bottleneck_channels = int(self.feature_channels[-1])
        self.disease_head = self._latent_head(bottleneck_channels, self.latent_dim)
        self.context_head = self._latent_head(bottleneck_channels, self.latent_dim)
        self.treatment_head = self._latent_head(bottleneck_channels, self.latent_dim)
        self.modulator = LatentFeatureModulator(self.feature_channels, self.latent_dim, modulation_scale=modulation_scale)
        self.causal_residual_head = nn.Conv3d(self.feature_channels[0], num_classes, kernel_size=1)
        self.causal_residual_gate = nn.Linear(int(latent_dim) * 2, num_classes)
        refiner_channels = max(8, self.feature_channels[0] // 2)
        self.spatial_disease_head = nn.Sequential(
            nn.Conv3d(self.feature_channels[0], refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, 1, kernel_size=1),
        )
        self.spatial_region_head = nn.Sequential(
            nn.Conv3d(self.feature_channels[0], refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, 3, kernel_size=1),
        )
        self.semantic_projector = nn.Sequential(
            nn.Conv3d(self.feature_channels[0], refiner_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, self.prototype_dim, kernel_size=1),
        )
        self.semantic_prototypes = nn.Parameter(torch.empty(4, self.prototype_dim))
        nn.init.normal_(self.semantic_prototypes, mean=0.0, std=0.02)
        self.register_buffer("category_confounders", torch.zeros(num_classes, self.feature_channels[0]))
        self.register_buffer("category_confounder_counts", torch.zeros(num_classes))
        self.boundary_head = nn.Sequential(
            nn.Conv3d(self.feature_channels[0], refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, 1, kernel_size=1),
        )
        self.modality_prior_head = nn.Sequential(
            nn.Conv3d(in_channels + num_classes + 1, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, num_classes, kernel_size=1),
        )
        self.logit_calibration_head = nn.Linear(int(latent_dim) * 2, num_classes * 2)
        frontdoor_region_channels = self.feature_channels[0] + num_classes + 3 + 1
        self.frontdoor_region_delta_head = nn.Sequential(
            nn.Conv3d(frontdoor_region_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, 3, kernel_size=1),
        )
        frontdoor_residual_channels = self.feature_channels[0] + num_classes + num_classes + 1
        self.frontdoor_residual_head = nn.Sequential(
            nn.Conv3d(frontdoor_residual_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, num_classes, kernel_size=1),
        )
        router_channels = self.feature_channels[0] + num_classes + num_classes + 3 + 3 + 3
        self.causal_mediator_router_head = nn.Sequential(
            nn.Conv3d(router_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, num_classes, kernel_size=1),
        )
        nested_condition_channels = self.feature_channels[0] + num_classes + 3 + 1 + 1 + 3
        self.nested_causal_condition_head = nn.Sequential(
            nn.Conv3d(nested_condition_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, 3, kernel_size=1),
        )
        nested_router_channels = self.feature_channels[0] + num_classes * 7 + 1
        self.nested_causal_router_head = nn.Sequential(
            nn.Conv3d(nested_router_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, num_classes, kernel_size=1),
        )
        region_causal_channels = self.feature_channels[0] * 2 + 3 + 3 + 1
        self.region_causal_head = nn.Sequential(
            nn.Conv3d(region_causal_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, num_classes, kernel_size=1),
        )
        cascade_input_channels = in_channels + num_classes * 4 + 2
        self.cascade_refiner_head = nn.Sequential(
            nn.Conv3d(cascade_input_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, refiner_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(refiner_channels, num_classes, kernel_size=1),
        )
        self.spatial_refiner_head = nn.Sequential(
            nn.Conv3d(
                self.feature_channels[0] + num_classes + 1 + 3 + 3 + 1,
                refiner_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv3d(refiner_channels, num_classes, kernel_size=1),
        )
        nn.init.zeros_(self.causal_residual_head.weight)
        nn.init.zeros_(self.causal_residual_head.bias)
        nn.init.zeros_(self.causal_residual_gate.weight)
        nn.init.zeros_(self.causal_residual_gate.bias)
        nn.init.zeros_(self.modality_prior_head[-1].weight)
        nn.init.zeros_(self.modality_prior_head[-1].bias)
        nn.init.zeros_(self.logit_calibration_head.weight)
        nn.init.zeros_(self.logit_calibration_head.bias)
        nn.init.zeros_(self.frontdoor_region_delta_head[-1].weight)
        nn.init.zeros_(self.frontdoor_region_delta_head[-1].bias)
        nn.init.zeros_(self.frontdoor_residual_head[-1].weight)
        nn.init.zeros_(self.frontdoor_residual_head[-1].bias)
        nn.init.zeros_(self.causal_mediator_router_head[-1].weight)
        nn.init.constant_(self.causal_mediator_router_head[-1].bias, -2.0)
        nn.init.zeros_(self.nested_causal_condition_head[-1].weight)
        nn.init.zeros_(self.nested_causal_condition_head[-1].bias)
        nn.init.zeros_(self.nested_causal_router_head[-1].weight)
        nn.init.constant_(self.nested_causal_router_head[-1].bias, -2.0)
        nn.init.zeros_(self.region_causal_head[-1].weight)
        nn.init.zeros_(self.region_causal_head[-1].bias)
        nn.init.zeros_(self.cascade_refiner_head[-1].weight)
        nn.init.zeros_(self.cascade_refiner_head[-1].bias)
        nn.init.zeros_(self.spatial_refiner_head[-1].weight)
        nn.init.zeros_(self.spatial_refiner_head[-1].bias)
        self.context_proxy_head = self._proxy_head(self.latent_dim, self.context_proxy_dim)
        self.disease_proxy_head = self._proxy_head(self.latent_dim, self.disease_proxy_dim)
        self.annotation_proxy_head = self._proxy_head(self.latent_dim, self.annotation_proxy_dim)
        self.context_from_disease_head = self._proxy_head(self.latent_dim, self.context_proxy_dim)
        self.disease_from_context_head = self._proxy_head(self.latent_dim, self.disease_proxy_dim)
        self.region_volume_head = self._proxy_head(self.latent_dim, 3)
        self.region_from_context_head = self._proxy_head(self.latent_dim, 3)
        self.sdd_context_teacher_head = self._mlp_head(self.latent_dim * 2, self.latent_dim, self.context_proxy_dim)
        self.sdd_region_teacher_head = self._mlp_head(self.latent_dim * 2, self.latent_dim, 3)
        self.sdd_treatment_joint_head = self._mlp_head(self.latent_dim * 2, self.latent_dim, self.treatment_proxy_dim)
        self.sdd_treatment_z_head = self._mlp_head(self.latent_dim, self.latent_dim, self.treatment_proxy_dim)
        self.sdd_treatment_c_head = self._mlp_head(self.latent_dim, self.latent_dim, self.treatment_proxy_dim)
        self.sdd_outcome_joint_head = self._mlp_head(self.latent_dim * 2, self.latent_dim, 3)
        self.sdd_outcome_y_head = self._mlp_head(self.latent_dim, self.latent_dim, 3)
        self.sdd_outcome_c_head = self._mlp_head(self.latent_dim, self.latent_dim, 3)
        self.cite_projector = self._mlp_head(self.latent_dim * 3, self.contrastive_dim, self.contrastive_dim)

    @staticmethod
    def _logit(probability: Tensor, eps: float = 1e-4) -> Tensor:
        probability = probability.clamp(eps, 1.0 - eps)
        return torch.log(probability) - torch.log1p(-probability)

    @classmethod
    def _region_logits_to_subregion_prior(cls, region_logits: Tensor) -> Tensor:
        region_prob = torch.sigmoid(region_logits)
        wt = region_prob[:, 0:1]
        tc = region_prob[:, 1:2]
        et = region_prob[:, 2:3]
        ncr_net = (tc * (1.0 - et)).clamp(0.0, 1.0)
        edema = (wt * (1.0 - tc)).clamp(0.0, 1.0)
        subregion_prob = torch.cat([ncr_net, edema, et], dim=1)
        return cls._logit(subregion_prob)

    @staticmethod
    def _subregion_prob_to_region_prob(subregion_prob: Tensor) -> Tensor:
        ncr_net = subregion_prob[:, 0:1]
        edema = subregion_prob[:, 1:2]
        enhancing = subregion_prob[:, 2:3]
        whole_tumor = 1.0 - (1.0 - ncr_net) * (1.0 - edema) * (1.0 - enhancing)
        tumor_core = 1.0 - (1.0 - ncr_net) * (1.0 - enhancing)
        return torch.cat([whole_tumor, tumor_core, enhancing], dim=1).clamp(0.0, 1.0)

    @classmethod
    def _nested_condition_logits_to_outputs(
        cls,
        raw_region_logits: Tensor,
        condition_delta: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert WT, TC|WT, ET|TC condition logits into valid BraTS subregions."""
        raw_region_prob = torch.sigmoid(raw_region_logits)
        raw_wt = raw_region_prob[:, 0:1]
        raw_tc = raw_region_prob[:, 1:2]
        raw_et = raw_region_prob[:, 2:3]
        raw_tc_given_wt = (raw_tc / raw_wt.clamp_min(1e-4)).clamp(0.0, 1.0)
        raw_et_given_tc = (raw_et / raw_tc.clamp_min(1e-4)).clamp(0.0, 1.0)
        raw_condition_logits = torch.cat(
            [
                cls._logit(raw_wt),
                cls._logit(raw_tc_given_wt),
                cls._logit(raw_et_given_tc),
            ],
            dim=1,
        )
        condition_logits = raw_condition_logits + condition_delta
        wt_prob = torch.sigmoid(condition_logits[:, 0:1])
        tc_prob = wt_prob * torch.sigmoid(condition_logits[:, 1:2])
        et_prob = tc_prob * torch.sigmoid(condition_logits[:, 2:3])
        region_prob = torch.cat([wt_prob, tc_prob, et_prob], dim=1).clamp(0.0, 1.0)
        ncr_net_prob = (1.0 - (1.0 - tc_prob) / (1.0 - et_prob).clamp_min(1e-4)).clamp(0.0, 1.0)
        edema_prob = (1.0 - (1.0 - wt_prob) / (1.0 - tc_prob).clamp_min(1e-4)).clamp(0.0, 1.0)
        subregion_prob = torch.cat([ncr_net_prob, edema_prob, et_prob], dim=1).clamp(0.0, 1.0)
        return condition_logits, cls._logit(region_prob), cls._logit(subregion_prob)

    def _prototype_logits(self, high_res_features: Tensor) -> Tensor:
        embedding = self.semantic_projector(high_res_features)
        embedding = F.normalize(embedding, dim=1)
        scale = 1.0 / self.prototype_temperature
        prototypes = F.normalize(self.semantic_prototypes, dim=1)
        return torch.einsum("bcdhw,kc->bkdhw", embedding, prototypes) * scale

    def set_category_confounders(self, values: Tensor, counts: Tensor | None = None) -> None:
        if tuple(values.shape) != tuple(self.category_confounders.shape):
            raise ValueError(
                "category confounder shape mismatch: "
                f"expected {tuple(self.category_confounders.shape)}, got {tuple(values.shape)}"
            )
        self.category_confounders.copy_(values.to(device=self.category_confounders.device, dtype=self.category_confounders.dtype))
        if counts is None:
            counts = torch.ones(self.category_confounder_counts.shape, device=self.category_confounder_counts.device)
        self.category_confounder_counts.copy_(
            counts.to(device=self.category_confounder_counts.device, dtype=self.category_confounder_counts.dtype)
        )

    def reset_category_confounders(self) -> None:
        self.category_confounders.zero_()
        self.category_confounder_counts.zero_()

    def _category_confounder_logits(self, high_res_features: Tensor) -> Tensor | None:
        if self.category_confounder_scale <= 0.0:
            return None
        valid = self.category_confounder_counts > 0
        if not bool(valid.any().detach().cpu()):
            return None
        features = F.normalize(high_res_features, dim=1)
        confounders = F.normalize(self.category_confounders.to(device=features.device, dtype=features.dtype), dim=1)
        logits = torch.einsum("bcdhw,kc->bkdhw", features, confounders)
        logits = logits / self.category_confounder_temperature
        valid_mask = valid.to(device=features.device).view(1, -1, 1, 1, 1)
        return torch.where(valid_mask, logits, torch.zeros_like(logits))

    def _category_confounder_expectation(self, high_res_features: Tensor) -> tuple[Tensor, Tensor]:
        valid = self.category_confounder_counts > 0
        zeros = torch.zeros_like(high_res_features)
        if not bool(valid.any().detach().cpu()):
            empty_logits = high_res_features.new_zeros(
                high_res_features.shape[0],
                self.num_classes,
                *high_res_features.shape[-3:],
            )
            return zeros, empty_logits
        features = F.normalize(high_res_features, dim=1)
        confounders = self.category_confounders.to(device=features.device, dtype=features.dtype)
        normalized_confounders = F.normalize(confounders, dim=1)
        logits = torch.einsum("bcdhw,kc->bkdhw", features, normalized_confounders)
        logits = logits / self.category_confounder_temperature
        valid_mask = valid.to(device=features.device).view(1, -1, 1, 1, 1)
        masked_logits = torch.where(valid_mask, logits, torch.full_like(logits, -1.0e4))
        weights = torch.softmax(masked_logits, dim=1)
        expected = torch.einsum("bkdhw,kc->bcdhw", weights, confounders)
        return expected, torch.where(valid_mask, logits, torch.zeros_like(logits))

    @staticmethod
    def _latent_head(in_features: int, latent_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    @staticmethod
    def _proxy_head(latent_dim: int, out_features: int) -> nn.Module | None:
        if out_features <= 0:
            return None
        return CausalMedNeXt._mlp_head(latent_dim, latent_dim, out_features)

    @staticmethod
    def _mlp_head(in_features: int, hidden_features: int, out_features: int) -> nn.Module | None:
        if out_features <= 0:
            return None
        return nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def encode_features(self, x: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.backbone.encode_features(x))

    def encode_latents(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        bottleneck = features[-1].mean(dim=(2, 3, 4))
        return self.disease_head(bottleneck), self.context_head(bottleneck)

    def encode_sdd_latents(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        bottleneck = features[-1].mean(dim=(2, 3, 4))
        z_y = self.disease_head(bottleneck)
        z_c = self.context_head(bottleneck)
        z_t = self.treatment_head(bottleneck)
        return z_y, z_c, z_t

    def treatment_propensity(self, z_t: Tensor, z_c: Tensor) -> Tensor | None:
        if self.sdd_treatment_joint_head is None:
            return None
        logits = self.sdd_treatment_joint_head(torch.cat([z_t, z_c], dim=1))
        if logits.shape[1] < 2:
            return torch.sigmoid(logits[:, 0])
        return torch.softmax(logits, dim=1)[:, 1]

    def _segment_outputs_from_latents(
        self,
        features: Sequence[Tensor],
        z_d: Tensor,
        z_c: Tensor,
        image: Tensor | None = None,
    ) -> dict[str, Tensor]:
        modulated = self.modulator(features, z_d, z_c)
        output_shape = tuple(int(value) for value in features[0].shape[-3:])
        decoder_features = self.backbone.decode_feature_maps(tuple(modulated))
        logits = self.backbone.logits_from_decoder_features(decoder_features, output_shape=output_shape)
        if not isinstance(logits, Tensor):
            raise TypeError("Causal MedNeXt backbone must be built without deep supervision.")
        high_res_features = decoder_features[0]
        if tuple(high_res_features.shape[-3:]) != output_shape:
            high_res_features = F.interpolate(high_res_features, size=output_shape, mode="trilinear", align_corners=False)
        residual = self.causal_residual_head(high_res_features)
        if tuple(residual.shape[-3:]) != output_shape:
            residual = F.interpolate(residual, size=output_shape, mode="trilinear", align_corners=False)
        gate = self.causal_residual_gate(torch.cat([z_d, z_c], dim=1))
        gate = 1.0 + 0.1 * torch.tanh(gate).view(gate.shape[0], gate.shape[1], 1, 1, 1)
        logits = logits + self.causal_residual_scale * residual * gate

        disease_attention_logits = self.spatial_disease_head(high_res_features)
        spatial_region_logits = self.spatial_region_head(high_res_features)
        prototype_logits = self._prototype_logits(high_res_features)
        boundary_logits = self.boundary_head(high_res_features)
        if tuple(disease_attention_logits.shape[-3:]) != output_shape:
            disease_attention_logits = F.interpolate(disease_attention_logits, size=output_shape, mode="trilinear", align_corners=False)
        if tuple(spatial_region_logits.shape[-3:]) != output_shape:
            spatial_region_logits = F.interpolate(spatial_region_logits, size=output_shape, mode="trilinear", align_corners=False)
        if tuple(prototype_logits.shape[-3:]) != output_shape:
            prototype_logits = F.interpolate(prototype_logits, size=output_shape, mode="trilinear", align_corners=False)
        if tuple(boundary_logits.shape[-3:]) != output_shape:
            boundary_logits = F.interpolate(boundary_logits, size=output_shape, mode="trilinear", align_corners=False)
        disease_attention = torch.sigmoid(disease_attention_logits)
        region_prob = torch.sigmoid(spatial_region_logits)
        subregion_prior_logits = self._region_logits_to_subregion_prior(spatial_region_logits)
        prototype_subregion_logits = prototype_logits[:, 1:] - prototype_logits[:, 0:1]
        category_confounder_logits = self._category_confounder_logits(high_res_features)
        if self.region_fusion_scale > 0.0:
            logits = logits + self.region_fusion_scale * disease_attention * subregion_prior_logits
        if self.prototype_fusion_scale > 0.0:
            logits = logits + self.prototype_fusion_scale * disease_attention * prototype_subregion_logits
        if category_confounder_logits is not None:
            logits = logits + self.category_confounder_scale * disease_attention * category_confounder_logits
        factual_region_prob = self._subregion_prob_to_region_prob(torch.sigmoid(logits.detach()))
        if self.region_causal_mask_source == "factual":
            wt_region_mask = factual_region_prob[:, :1]
        else:
            wt_region_mask = region_prob[:, :1]
        region_causal_mask = self.region_causal_background_leak + (1.0 - self.region_causal_background_leak) * wt_region_mask
        region_causal_features = high_res_features * region_causal_mask
        expected_confounder, category_attention_logits = self._category_confounder_expectation(region_causal_features)
        region_causal_input = torch.cat(
            [
                region_causal_features,
                expected_confounder,
                spatial_region_logits,
                region_prob,
                disease_attention,
            ],
            dim=1,
        )
        region_causal_delta = self.region_causal_head(region_causal_input)
        region_causal_base_logits = logits if self.region_causal_base == "factual" else subregion_prior_logits
        region_causal_logits = region_causal_base_logits + region_causal_delta
        if self.region_causal_bottleneck_scale > 0.0:
            keep_raw = 1.0 - self.region_causal_bottleneck_scale
            logits = keep_raw * logits + self.region_causal_bottleneck_scale * region_causal_logits
        if image is not None:
            modality_image = image
            if tuple(modality_image.shape[-3:]) != output_shape:
                modality_image = F.interpolate(modality_image, size=output_shape, mode="trilinear", align_corners=False)
            modality_prior_input = torch.cat([modality_image, logits, disease_attention], dim=1)
            modality_prior_logits = self.modality_prior_head(modality_prior_input)
            if tuple(modality_prior_logits.shape[-3:]) != output_shape:
                modality_prior_logits = F.interpolate(
                    modality_prior_logits,
                    size=output_shape,
                    mode="trilinear",
                    align_corners=False,
                )
            if self.modality_prior_scale > 0.0:
                logits = logits + self.modality_prior_scale * (0.25 + disease_attention) * modality_prior_logits
        else:
            modality_prior_logits = torch.zeros_like(logits)
        calibration = self.logit_calibration_head(torch.cat([z_d, z_c], dim=1))
        calibration_scale, calibration_bias = calibration.chunk(2, dim=1)
        calibration_scale = 1.0 + 0.1 * self.logit_calibration_scale * torch.tanh(calibration_scale)
        calibration_bias = self.logit_calibration_scale * calibration_bias
        logits = logits * calibration_scale.view(calibration_scale.shape[0], calibration_scale.shape[1], 1, 1, 1)
        logits = logits + calibration_bias.view(calibration_bias.shape[0], calibration_bias.shape[1], 1, 1, 1)
        frontdoor_base_logits = logits
        base_region_prob = self._subregion_prob_to_region_prob(torch.sigmoid(frontdoor_base_logits))
        frontdoor_raw_region_logits = self._logit(base_region_prob)
        frontdoor_region_input = torch.cat(
            [high_res_features, frontdoor_base_logits, base_region_prob, disease_attention],
            dim=1,
        )
        frontdoor_region_delta = self.frontdoor_region_delta_head(frontdoor_region_input)
        frontdoor_region_logits = frontdoor_raw_region_logits + frontdoor_region_delta
        frontdoor_subregion_logits = self._region_logits_to_subregion_prior(frontdoor_region_logits)
        frontdoor_subregion_prob = torch.sigmoid(frontdoor_subregion_logits)
        frontdoor_residual_input = torch.cat(
            [high_res_features, frontdoor_base_logits, frontdoor_subregion_prob, disease_attention],
            dim=1,
        )
        frontdoor_residual_delta = self.frontdoor_residual_head(frontdoor_residual_input)
        frontdoor_logits = frontdoor_subregion_logits + self.frontdoor_residual_scale * disease_attention * frontdoor_residual_delta
        frontdoor_disagreement = (torch.sigmoid(frontdoor_logits) - torch.sigmoid(frontdoor_base_logits)).abs()
        frontdoor_uncertainty = (4.0 * torch.sigmoid(frontdoor_base_logits) * (1.0 - torch.sigmoid(frontdoor_base_logits))).clamp(0.0, 1.0)
        router_input = torch.cat(
            [
                high_res_features,
                frontdoor_base_logits,
                frontdoor_subregion_prob,
                base_region_prob,
                frontdoor_disagreement,
                frontdoor_uncertainty,
            ],
            dim=1,
        )
        causal_mediator_router_logits = self.causal_mediator_router_head(router_input)
        causal_mediator_router_gate = torch.sigmoid(causal_mediator_router_logits) * disease_attention
        if self.frontdoor_mediator_scale > 0.0:
            keep_raw = 1.0 - self.frontdoor_mediator_scale
            logits = keep_raw * logits + self.frontdoor_mediator_scale * frontdoor_logits
        if self.use_causal_mediator_router:
            logits = logits + causal_mediator_router_gate * (frontdoor_logits - logits)
        nested_causal_base_logits = logits
        nested_base_prob = torch.sigmoid(nested_causal_base_logits)
        nested_base_region_prob = self._subregion_prob_to_region_prob(nested_base_prob)
        nested_raw_region_logits = self._logit(nested_base_region_prob)
        nested_condition_input = torch.cat(
            [
                high_res_features,
                nested_causal_base_logits,
                nested_base_region_prob,
                disease_attention,
                torch.sigmoid(boundary_logits),
                category_attention_logits,
            ],
            dim=1,
        )
        nested_condition_delta = self.nested_causal_condition_head(nested_condition_input)
        (
            nested_causal_condition_logits,
            nested_causal_region_logits,
            nested_causal_subregion_logits,
        ) = self._nested_condition_logits_to_outputs(nested_raw_region_logits, nested_condition_delta)
        nested_causal_subregion_prob = torch.sigmoid(nested_causal_subregion_logits)
        nested_causal_region_prob = torch.sigmoid(nested_causal_region_logits)
        nested_causal_disagreement = (nested_causal_subregion_prob - nested_base_prob).abs()
        nested_causal_uncertainty = (4.0 * nested_base_prob * (1.0 - nested_base_prob)).clamp(0.0, 1.0)
        nested_router_input = torch.cat(
            [
                high_res_features,
                nested_causal_base_logits,
                nested_base_prob,
                nested_base_region_prob,
                nested_causal_subregion_prob,
                nested_causal_region_prob,
                nested_causal_disagreement,
                nested_causal_uncertainty,
                disease_attention,
            ],
            dim=1,
        )
        nested_causal_router_logits = self.nested_causal_router_head(nested_router_input)
        nested_causal_router_gate = (
            self.nested_causal_gate_scale * torch.sigmoid(nested_causal_router_logits) * disease_attention
        )
        if self.use_nested_causal_intervention:
            logits = logits + nested_causal_router_gate * (nested_causal_subregion_logits - logits)
        cascade_base_logits = logits
        if image is not None:
            rough_prob = torch.sigmoid(cascade_base_logits)
            rough_uncertainty = (4.0 * rough_prob * (1.0 - rough_prob)).clamp(0.0, 1.0)
            rough_region_prob = self._subregion_prob_to_region_prob(rough_prob)
            rough_foreground_prob = rough_prob.amax(dim=1, keepdim=True)
            cascade_input = torch.cat(
                [
                    modality_image,
                    cascade_base_logits,
                    rough_prob,
                    rough_uncertainty,
                    rough_region_prob,
                    disease_attention,
                    rough_foreground_prob,
                ],
                dim=1,
            )
            cascade_delta = self.cascade_refiner_head(cascade_input)
            cascade_gate = (0.25 + disease_attention + rough_uncertainty.mean(dim=1, keepdim=True)).clamp(0.0, 2.0)
            if self.cascade_refiner_scale > 0.0:
                logits = cascade_base_logits + self.cascade_refiner_scale * cascade_gate * cascade_delta
        else:
            rough_prob = torch.sigmoid(cascade_base_logits)
            rough_uncertainty = torch.zeros_like(cascade_base_logits)
            cascade_delta = torch.zeros_like(cascade_base_logits)
        cascade_logits = logits
        boundary_prob = torch.sigmoid(boundary_logits)
        prototype_prob = torch.sigmoid(prototype_subregion_logits)
        refiner_input = torch.cat(
            [high_res_features, logits, disease_attention, region_prob, prototype_prob, boundary_prob],
            dim=1,
        )
        refiner_delta = self.spatial_refiner_head(refiner_input)
        lesion_gate = 0.25 + disease_attention
        refined_logits = logits + self.spatial_refiner_scale * refiner_delta * lesion_gate
        outputs = {
            "logits": refined_logits,
            "pre_refiner_logits": logits,
            "disease_attention_logits": disease_attention_logits,
            "spatial_region_logits": spatial_region_logits,
            "subregion_prior_logits": subregion_prior_logits,
            "prototype_logits": prototype_logits,
            "prototype_subregion_logits": prototype_subregion_logits,
            "category_confounder_logits": torch.zeros_like(logits) if category_confounder_logits is None else category_confounder_logits,
            "category_confounder_attention_logits": category_attention_logits,
            "region_causal_mask": region_causal_mask,
            "region_causal_base_logits": region_causal_base_logits,
            "region_causal_delta": region_causal_delta,
            "region_causal_logits": region_causal_logits,
            "boundary_logits": boundary_logits,
            "modality_prior_logits": modality_prior_logits,
            "logit_calibration_scale": calibration_scale,
            "logit_calibration_bias": calibration_bias,
            "frontdoor_base_logits": frontdoor_base_logits,
            "frontdoor_raw_region_logits": frontdoor_raw_region_logits,
            "frontdoor_region_logits": frontdoor_region_logits,
            "frontdoor_region_delta": frontdoor_region_delta,
            "frontdoor_subregion_logits": frontdoor_subregion_logits,
            "frontdoor_residual_delta": frontdoor_residual_delta,
            "frontdoor_logits": frontdoor_logits,
            "causal_mediator_router_logits": causal_mediator_router_logits,
            "causal_mediator_router_gate": causal_mediator_router_gate,
            "nested_causal_base_logits": nested_causal_base_logits,
            "nested_causal_raw_region_logits": nested_raw_region_logits,
            "nested_causal_condition_delta": nested_condition_delta,
            "nested_causal_condition_logits": nested_causal_condition_logits,
            "nested_causal_region_logits": nested_causal_region_logits,
            "nested_causal_subregion_logits": nested_causal_subregion_logits,
            "nested_causal_router_logits": nested_causal_router_logits,
            "nested_causal_router_gate": nested_causal_router_gate,
            "cascade_base_logits": cascade_base_logits,
            "cascade_refiner_delta": cascade_delta,
            "cascade_uncertainty": rough_uncertainty,
            "cascade_logits": cascade_logits,
            "causal_refiner_delta": refiner_delta,
            "causal_high_res_features": high_res_features,
        }
        return outputs

    def segment_from_latents(self, features: Sequence[Tensor], z_d: Tensor, z_c: Tensor, image: Tensor | None = None) -> Tensor:
        return self._segment_outputs_from_latents(features, z_d, z_c, image=image)["logits"]

    def predict_proxies(self, z_d: Tensor, z_c: Tensor, z_t: Tensor, adversary_strength: float = 1.0) -> dict[str, Tensor]:
        predictions: dict[str, Tensor] = {}
        disease_context = torch.cat([z_d, z_c], dim=1)
        treatment_context = torch.cat([z_t, z_c], dim=1)
        sdd_full = torch.cat([z_t, z_c, z_d], dim=1)
        if self.context_proxy_head is not None:
            predictions["context_proxy_logits"] = self.context_proxy_head(z_c)
        if self.disease_proxy_head is not None:
            predictions["disease_proxy_logits"] = self.disease_proxy_head(z_d)
        if self.annotation_proxy_head is not None:
            predictions["annotation_proxy_logits"] = self.annotation_proxy_head(z_c)
        if self.context_from_disease_head is not None:
            predictions["context_from_disease_logits"] = self.context_from_disease_head(
                gradient_reverse(z_d, adversary_strength)
            )
        if self.disease_from_context_head is not None:
            predictions["disease_from_context_logits"] = self.disease_from_context_head(
                gradient_reverse(z_c, adversary_strength)
            )
        if self.region_volume_head is not None:
            predictions["region_volume_logits"] = self.region_volume_head(z_d)
        if self.region_from_context_head is not None:
            predictions["region_from_context_logits"] = self.region_from_context_head(
                gradient_reverse(z_c, adversary_strength)
            )
        if self.sdd_context_teacher_head is not None:
            predictions["sdd_context_teacher_logits"] = self.sdd_context_teacher_head(disease_context)
        if self.sdd_region_teacher_head is not None:
            predictions["sdd_region_teacher_logits"] = self.sdd_region_teacher_head(disease_context)
        if self.sdd_treatment_joint_head is not None:
            predictions["sdd_treatment_joint_logits"] = self.sdd_treatment_joint_head(treatment_context)
            predictions["sdd_treatment_z_logits"] = self.sdd_treatment_z_head(z_t)
            predictions["sdd_treatment_c_logits"] = self.sdd_treatment_c_head(z_c)
        if self.sdd_outcome_joint_head is not None:
            predictions["sdd_outcome_joint_logits"] = self.sdd_outcome_joint_head(disease_context)
            predictions["sdd_outcome_y_logits"] = self.sdd_outcome_y_head(z_d)
            predictions["sdd_outcome_c_logits"] = self.sdd_outcome_c_head(z_c)
        if self.cite_projector is not None:
            predictions["cite_anchor"] = self.cite_projector(sdd_full)
        return predictions

    def backdoor_adjusted_logits(
        self,
        features: Sequence[Tensor],
        z_d: Tensor,
        context_bank: Tensor,
        max_contexts: int | None = None,
        image: Tensor | None = None,
    ) -> Tensor:
        if context_bank.ndim != 2:
            raise ValueError(f"context_bank must have shape [K, latent_dim], got {tuple(context_bank.shape)}")
        if context_bank.shape[1] != z_d.shape[1]:
            raise ValueError(f"context_bank latent dim {context_bank.shape[1]} does not match z_d dim {z_d.shape[1]}")
        bank = context_bank.to(device=z_d.device, dtype=z_d.dtype)
        if max_contexts is not None and max_contexts > 0 and bank.shape[0] > max_contexts:
            positions = torch.linspace(0, bank.shape[0] - 1, steps=max_contexts, device=bank.device)
            bank = bank[positions.round().long()]
        logits = []
        for context in bank:
            z_c = context.unsqueeze(0).expand(z_d.shape[0], -1)
            logits.append(self.segment_from_latents(features, z_d, z_c, image=image))
        return torch.stack(logits, dim=0).mean(dim=0)

    def forward(
        self,
        x: Tensor,
        context_bank: Tensor | None = None,
        max_adjustment_contexts: int | None = None,
        adversary_strength: float = 1.0,
        max_contrastive_negatives: int | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        features = self.encode_features(x)
        z_d, z_c, z_t = self.encode_sdd_latents(features)
        segmentation_outputs = self._segment_outputs_from_latents(features, z_d, z_c, image=x)
        logits = segmentation_outputs["logits"]
        outputs: dict[str, Tensor | tuple[Tensor, ...]] = {
            "logits": logits,
            "z_d": z_d,
            "z_c": z_c,
            "z_t": z_t,
            "features": features,
        }
        outputs.update(segmentation_outputs)
        outputs.update(self.predict_proxies(z_d, z_c, z_t, adversary_strength=adversary_strength))
        if context_bank is not None:
            outputs["adjusted_logits"] = self.backdoor_adjusted_logits(
                features,
                z_d,
                context_bank,
                max_contexts=max_adjustment_contexts,
                image=x,
            )
        return outputs

    def add_cite_outputs(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        contrastive_bank: dict[str, Tensor] | None,
        max_negatives: int | None = None,
    ) -> None:
        if contrastive_bank is None or self.cite_projector is None or "cite_anchor" not in outputs:
            return
        anchor = outputs["cite_anchor"]
        if not isinstance(anchor, Tensor):
            return
        z_t = contrastive_bank["z_t"].to(device=anchor.device, dtype=anchor.dtype)
        z_c = contrastive_bank["z_c"].to(device=anchor.device, dtype=anchor.dtype)
        z_d = contrastive_bank["z_d"].to(device=anchor.device, dtype=anchor.dtype)
        scores = contrastive_bank["propensity"].to(device=anchor.device, dtype=anchor.dtype).view(-1)
        if z_t.numel() == 0 or z_c.numel() == 0 or z_d.numel() == 0 or scores.numel() == 0:
            return
        bank_rep = torch.cat([z_t, z_c, z_d], dim=1)
        projected = self.cite_projector(bank_rep)
        if not isinstance(projected, Tensor) or projected.ndim != 2:
            return
        requested = max(1, int(max_negatives or 16))
        positive_count = min(projected.shape[0], requested)
        positive_pool = torch.argsort(torch.abs(scores - 0.5))[:positive_count]
        if self.training and positive_pool.numel() > 1:
            choice = positive_pool[torch.randint(positive_pool.numel(), (anchor.shape[0],), device=anchor.device)]
        else:
            choice = positive_pool[torch.arange(anchor.shape[0], device=anchor.device) % positive_pool.numel()]
        low = torch.argsort(scores)
        high = torch.argsort(scores, descending=True)
        extreme_pool = torch.cat([low[:requested], high[:requested]], dim=0).unique()
        if extreme_pool.numel() > requested:
            if self.training:
                extreme_pool = extreme_pool[torch.randperm(extreme_pool.numel(), device=anchor.device)[:requested]]
            else:
                extreme_pool = extreme_pool[:requested]
        negative_indices = extreme_pool
        outputs["cite_positive"] = projected[choice]
        outputs["cite_negative"] = projected[negative_indices]
        if "treatment_label" in contrastive_bank:
            outputs["sdd_bank_z_d"] = z_d.detach()
            outputs["sdd_bank_treatment_label"] = contrastive_bank["treatment_label"].to(anchor.device).view(-1).long()

    def load_baseline_state_dict(self, state_dict: dict[str, Tensor], strict_backbone: bool = True) -> None:
        if any(str(key).startswith("backbone.") for key in state_dict):
            state_dict = {
                str(key).removeprefix("backbone."): value
                for key, value in state_dict.items()
                if str(key).startswith("backbone.")
            }
        self.backbone.load_state_dict(state_dict, strict=strict_backbone)

    def load_compatible_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, list[str]]:
        current = self.state_dict()
        compatible: dict[str, Tensor] = {}
        skipped: list[str] = []
        unexpected: list[str] = []
        for key, value in state_dict.items():
            if key not in current:
                unexpected.append(str(key))
                continue
            if tuple(current[key].shape) != tuple(value.shape):
                skipped.append(str(key))
                continue
            compatible[str(key)] = value
        result = self.load_state_dict(compatible, strict=False)
        return {
            "missing_keys": [str(key) for key in result.missing_keys],
            "unexpected_keys": unexpected + [str(key) for key in result.unexpected_keys],
            "skipped_shape_keys": skipped,
        }


def build_causal_mednext(
    model_id: str = "S",
    kernel_size: int = 3,
    latent_dim: int = 128,
    num_classes: int = 3,
    context_proxy_dim: int = 0,
    disease_proxy_dim: int = 0,
    annotation_proxy_dim: int = 0,
    treatment_proxy_dim: int = 2,
    base_channels: int | None = None,
    modulation_scale: float = 0.1,
    causal_residual_scale: float = 0.2,
    contrastive_dim: int = 64,
    spatial_refiner_scale: float = 0.5,
    region_fusion_scale: float = 0.0,
    prototype_dim: int = 32,
    prototype_fusion_scale: float = 0.0,
    prototype_temperature: float = 0.1,
    category_confounder_scale: float = 0.0,
    category_confounder_temperature: float = 0.2,
    modality_prior_scale: float = 0.0,
    logit_calibration_scale: float = 0.0,
    cascade_refiner_scale: float = 0.0,
    frontdoor_mediator_scale: float = 0.0,
    frontdoor_residual_scale: float = 0.25,
    use_causal_mediator_router: bool = False,
    use_nested_causal_intervention: bool = False,
    nested_causal_gate_scale: float = 1.0,
    region_causal_bottleneck_scale: float = 0.0,
    region_causal_background_leak: float = 0.05,
    region_causal_base: str = "prior",
    region_causal_mask_source: str = "spatial",
) -> CausalMedNeXt:
    return CausalMedNeXt(
        model_id=model_id,
        kernel_size=kernel_size,
        in_channels=4,
        num_classes=num_classes,
        latent_dim=latent_dim,
        context_proxy_dim=context_proxy_dim,
        disease_proxy_dim=disease_proxy_dim,
        annotation_proxy_dim=annotation_proxy_dim,
        treatment_proxy_dim=treatment_proxy_dim,
        base_channels=base_channels,
        modulation_scale=modulation_scale,
        causal_residual_scale=causal_residual_scale,
        contrastive_dim=contrastive_dim,
        spatial_refiner_scale=spatial_refiner_scale,
        region_fusion_scale=region_fusion_scale,
        prototype_dim=prototype_dim,
        prototype_fusion_scale=prototype_fusion_scale,
        prototype_temperature=prototype_temperature,
        category_confounder_scale=category_confounder_scale,
        category_confounder_temperature=category_confounder_temperature,
        modality_prior_scale=modality_prior_scale,
        logit_calibration_scale=logit_calibration_scale,
        cascade_refiner_scale=cascade_refiner_scale,
        frontdoor_mediator_scale=frontdoor_mediator_scale,
        frontdoor_residual_scale=frontdoor_residual_scale,
        use_causal_mediator_router=use_causal_mediator_router,
        use_nested_causal_intervention=use_nested_causal_intervention,
        nested_causal_gate_scale=nested_causal_gate_scale,
        region_causal_bottleneck_scale=region_causal_bottleneck_scale,
        region_causal_background_leak=region_causal_background_leak,
        region_causal_base=region_causal_base,
        region_causal_mask_source=region_causal_mask_source,
    )
