from __future__ import annotations

import torch

from baselines.mednext.common import main_logits, segmentation_loss
from baselines.mednext.causal import CausalMedNeXt, build_causal_mednext
from baselines.mednext.evaluate_causal_utsw import _apply_nonenhancing_core_completion, _fuse_style_tta_logits
from baselines.mednext.model import build_mednext_segmenter
from baselines.mednext.roi_refiner import CausalRoiRefiner, bbox_from_mask, crop_resize_3d, paste_resized_3d, scale_bbox
from baselines.mednext.train_causal_utsw import LesionInterventionBank, apply_style_intervention
from baselines.segformer3d.train_causal_utsw import (
    _balanced_region_mediator_loss,
    _balanced_subregion_mediator_loss,
    _boundary_mediator_loss,
    _frontdoor_router_advantage_loss,
    _prototype_mediator_loss,
    _subregion_class_target,
)


def test_mednext_segmenter_forward_shape_tiny_channels() -> None:
    model = build_mednext_segmenter(model_id="S", kernel_size=3, base_channels=4, deep_supervision=True)
    image = torch.randn(1, 4, 32, 32, 32)
    output = model(image)
    logits = main_logits(output)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert isinstance(output, list)
    assert len(output) == 5


def test_mednext_deep_supervision_loss_supports_auxiliary_scales() -> None:
    output = [
        torch.randn(1, 3, 32, 32, 32),
        torch.randn(1, 3, 16, 16, 16),
        torch.randn(1, 3, 8, 8, 8),
    ]
    target = torch.randint(0, 2, (1, 3, 32, 32, 32)).float()
    loss = segmentation_loss(output, target, "1.0,0.5,0.25")
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_causal_mednext_outputs_adjusted_logits_and_latents() -> None:
    model = build_causal_mednext(
        model_id="S",
        kernel_size=3,
        base_channels=4,
        latent_dim=8,
        context_proxy_dim=2,
        disease_proxy_dim=3,
        annotation_proxy_dim=1,
        category_confounder_scale=0.2,
        use_causal_mediator_router=True,
        use_nested_causal_intervention=True,
        region_causal_bottleneck_scale=0.5,
        region_causal_base="factual",
    )
    model.set_category_confounders(torch.randn(3, model.feature_channels[0]), torch.ones(3))
    image = torch.randn(1, 4, 32, 32, 32)
    context_bank = torch.randn(2, 8)
    outputs = model(image, context_bank=context_bank, max_adjustment_contexts=2)
    assert outputs["logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["adjusted_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["pre_refiner_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["disease_attention_logits"].shape == (1, 1, 32, 32, 32)
    assert outputs["spatial_region_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["subregion_prior_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["prototype_logits"].shape == (1, 4, 32, 32, 32)
    assert outputs["prototype_subregion_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["category_confounder_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["category_confounder_attention_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["region_causal_mask"].shape == (1, 1, 32, 32, 32)
    assert outputs["region_causal_base_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["region_causal_delta"].shape == (1, 3, 32, 32, 32)
    assert outputs["region_causal_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["boundary_logits"].shape == (1, 1, 32, 32, 32)
    assert outputs["modality_prior_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["logit_calibration_scale"].shape == (1, 3)
    assert outputs["logit_calibration_bias"].shape == (1, 3)
    assert outputs["frontdoor_base_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["frontdoor_raw_region_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["frontdoor_region_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["frontdoor_region_delta"].shape == (1, 3, 32, 32, 32)
    assert outputs["frontdoor_subregion_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["frontdoor_residual_delta"].shape == (1, 3, 32, 32, 32)
    assert outputs["frontdoor_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["causal_mediator_router_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["causal_mediator_router_gate"].shape == (1, 3, 32, 32, 32)
    assert outputs["nested_causal_base_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["nested_causal_raw_region_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["nested_causal_condition_delta"].shape == (1, 3, 32, 32, 32)
    assert outputs["nested_causal_condition_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["nested_causal_region_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["nested_causal_subregion_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["nested_causal_router_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["nested_causal_router_gate"].shape == (1, 3, 32, 32, 32)
    assert outputs["cascade_base_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["cascade_refiner_delta"].shape == (1, 3, 32, 32, 32)
    assert outputs["cascade_uncertainty"].shape == (1, 3, 32, 32, 32)
    assert outputs["cascade_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["causal_refiner_delta"].shape == (1, 3, 32, 32, 32)
    assert outputs["z_d"].shape == (1, 8)
    assert outputs["z_c"].shape == (1, 8)
    assert outputs["z_t"].shape == (1, 8)
    assert outputs["context_proxy_logits"].shape == (1, 2)
    assert outputs["disease_proxy_logits"].shape == (1, 3)
    assert outputs["annotation_proxy_logits"].shape == (1, 1)
    assert outputs["context_from_disease_logits"].shape == (1, 2)
    assert outputs["disease_from_context_logits"].shape == (1, 3)
    assert outputs["region_volume_logits"].shape == (1, 3)
    assert outputs["region_from_context_logits"].shape == (1, 3)
    assert outputs["sdd_context_teacher_logits"].shape == (1, 2)
    assert outputs["sdd_region_teacher_logits"].shape == (1, 3)
    assert outputs["sdd_treatment_joint_logits"].shape == (1, 2)
    assert outputs["sdd_treatment_z_logits"].shape == (1, 2)
    assert outputs["sdd_treatment_c_logits"].shape == (1, 2)
    assert outputs["sdd_outcome_joint_logits"].shape == (1, 3)
    assert outputs["sdd_outcome_y_logits"].shape == (1, 3)
    assert outputs["sdd_outcome_c_logits"].shape == (1, 3)
    assert outputs["cite_anchor"].shape == (1, 64)
    model.add_cite_outputs(
        outputs,
        {
            "z_t": torch.randn(4, 8),
            "z_c": torch.randn(4, 8),
            "z_d": torch.randn(4, 8),
            "propensity": torch.tensor([0.05, 0.45, 0.55, 0.95]),
            "treatment_label": torch.tensor([0, 0, 1, 1]),
        },
        max_negatives=2,
    )
    assert outputs["cite_positive"].shape == (1, 64)
    assert outputs["cite_negative"].shape[1] == 64
    assert outputs["sdd_bank_z_d"].shape == (4, 8)
    assert outputs["sdd_bank_treatment_label"].shape == (4,)


def test_style_intervention_preserves_shape_and_changes_intensity() -> None:
    args = type(
        "Args",
        (),
        {
            "style_scale_range": "0.8,1.2",
            "style_shift_range": "-0.1,0.1",
            "style_gamma_range": "0.9,1.1",
            "style_bias_strength": 0.1,
            "style_bias_grid_size": 2,
            "style_noise_std": 0.01,
            "style_modality_dropout_prob": 0.0,
            "style_randconv_layers": 1,
            "style_randconv_kernel_size": 3,
            "style_randconv_strength": 0.5,
        },
    )()
    torch.manual_seed(7)
    image = torch.randn(2, 4, 16, 16, 16)
    augmented = apply_style_intervention(image, args)

    assert augmented.shape == image.shape
    assert torch.isfinite(augmented).all()
    assert not torch.allclose(augmented, image)


def test_region_logits_to_subregion_prior_respects_brats_hierarchy() -> None:
    logits = torch.full((1, 3, 1, 1, 3), -8.0)
    logits[:, 0, :, :, 0] = 8.0
    logits[:, 1, :, :, 1] = 8.0
    logits[:, 2, :, :, 2] = 8.0

    prior = torch.sigmoid(CausalMedNeXt._region_logits_to_subregion_prior(logits))

    assert prior[:, 1, :, :, 0] > 0.99
    assert prior[:, 0, :, :, 1] > 0.99
    assert prior[:, 2, :, :, 2] > 0.99


def test_nested_causal_outputs_respect_region_inclusion() -> None:
    raw_region_logits = torch.randn(2, 3, 4, 4, 4)
    condition_delta = torch.randn_like(raw_region_logits)

    _, nested_region_logits, nested_subregion_logits = CausalMedNeXt._nested_condition_logits_to_outputs(
        raw_region_logits,
        condition_delta,
    )
    nested_region = torch.sigmoid(nested_region_logits)
    nested_subregion = torch.sigmoid(nested_subregion_logits)

    assert torch.all(nested_region[:, 2] <= nested_region[:, 1] + 1e-5)
    assert torch.all(nested_region[:, 1] <= nested_region[:, 0] + 1e-5)
    reconstructed_wt = 1.0 - (1.0 - nested_subregion[:, 0]) * (1.0 - nested_subregion[:, 1]) * (1.0 - nested_subregion[:, 2])
    reconstructed_tc = 1.0 - (1.0 - nested_subregion[:, 0]) * (1.0 - nested_subregion[:, 2])
    assert torch.allclose(reconstructed_wt, nested_region[:, 0], atol=5e-4)
    assert torch.allclose(reconstructed_tc, nested_region[:, 1], atol=5e-4)
    assert torch.allclose(nested_subregion[:, 2], nested_region[:, 2], atol=5e-4)


def test_prototype_and_boundary_mediator_losses_are_finite() -> None:
    target = torch.zeros(1, 3, 8, 8, 8)
    target[:, 0, 1:3, 1:3, 1:3] = 1.0
    target[:, 1, 4:7, 4:7, 4:7] = 1.0
    target[:, 2, 3:5, 3:5, 3:5] = 1.0

    labels = _subregion_class_target(target)
    assert set(labels.unique().tolist()) == {0, 1, 2, 3}

    args = type("Args", (), {"channel_loss_weights": "1.0,1.0,1.0"})()
    prototype_logits = torch.randn(1, 4, 8, 8, 8)
    subregion_logits = prototype_logits[:, 1:] - prototype_logits[:, 0:1]
    prototype_loss = _prototype_mediator_loss(prototype_logits, subregion_logits, target, args)
    boundary_loss = _boundary_mediator_loss(torch.randn(1, 1, 8, 8, 8), target)

    assert prototype_loss is not None
    assert boundary_loss is not None
    assert torch.isfinite(prototype_loss)
    assert torch.isfinite(boundary_loss)


def test_balanced_frontdoor_mediator_losses_are_finite_on_sparse_masks() -> None:
    target = torch.zeros(1, 3, 8, 8, 8)
    target[:, 2, 2:4, 2:4, 2:4] = 1.0
    region_logits = torch.randn(1, 3, 8, 8, 8)
    subregion_logits = torch.randn(1, 3, 8, 8, 8)

    region_loss = _balanced_region_mediator_loss(region_logits, target)
    subregion_loss = _balanced_subregion_mediator_loss(subregion_logits, target)

    assert region_loss is not None
    assert subregion_loss is not None
    assert torch.isfinite(region_loss)
    assert torch.isfinite(subregion_loss)


def test_frontdoor_router_advantage_loss_is_finite_when_mediator_improves() -> None:
    target = torch.zeros(1, 3, 8, 8, 8)
    target[:, 2, 2:4, 2:4, 2:4] = 1.0
    base_logits = torch.full_like(target, -2.0)
    mediator_logits = base_logits.clone()
    mediator_logits[:, 2, 2:4, 2:4, 2:4] = 4.0
    router_logits = torch.zeros_like(target)

    loss = _frontdoor_router_advantage_loss(router_logits, base_logits, mediator_logits, target)

    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_balanced_subregion_loss_has_finite_gradients_for_confident_logits() -> None:
    target = torch.zeros(1, 3, 8, 8, 8)
    target[:, 2, 2:4, 2:4, 2:4] = 1.0
    logits = torch.where(target > 0.5, torch.full_like(target, 20.0), torch.full_like(target, -20.0))
    logits.requires_grad_(True)

    loss = _balanced_subregion_mediator_loss(logits, target)

    assert loss is not None
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_roi_refiner_helpers_preserve_baseline_at_initialization() -> None:
    image = torch.randn(4, 20, 24, 28)
    mask = torch.zeros(3, 20, 24, 28)
    mask[:, 6:12, 7:14, 8:18] = 1.0
    bbox = bbox_from_mask(mask, margin=2)
    scaled = scale_bbox(bbox, source_shape=(20, 24, 28), target_shape=(40, 48, 56))
    assert scaled[0][0] <= 8 and scaled[0][1] >= 24

    roi_image = crop_resize_3d(image, bbox, size=16, mode="trilinear")
    coarse = torch.randn(3, 16, 16, 16)
    refiner = CausalRoiRefiner(channels=4)
    refined = refiner(roi_image.unsqueeze(0), coarse.unsqueeze(0))
    assert refined.shape == (1, 3, 16, 16, 16)
    assert torch.allclose(refined, coarse.unsqueeze(0), atol=1e-6)

    pasted = paste_resized_3d(torch.zeros(3, 20, 24, 28), refined.squeeze(0), bbox)
    assert pasted.shape == (3, 20, 24, 28)
    assert pasted[:, bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]].abs().sum() > 0


def test_causal_mednext_compatible_loader_skips_shape_mismatch() -> None:
    model = build_causal_mednext(model_id="S", kernel_size=3, base_channels=4, latent_dim=8)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state["spatial_refiner_head.0.weight"] = torch.randn(1, 1, 1, 1, 1)

    report = model.load_compatible_state_dict(state)

    assert "spatial_refiner_head.0.weight" in report["skipped_shape_keys"]


def test_lesion_intervention_bank_pastes_and_erases_foreground() -> None:
    bank = LesionInterventionBank(
        max_patches=2,
        min_voxels=1,
        edge_softening=3,
        min_brain_coverage=0.5,
        match_recipient_moments=True,
    )
    image = torch.randn(1, 4, 16, 16, 16)
    target = torch.zeros(1, 3, 16, 16, 16)
    target[:, 2, 6:9, 6:9, 6:9] = 1.0
    bank.update(image, target)

    recipient = torch.ones_like(image)
    paste = bank.paste(recipient, torch.zeros_like(target))
    erase = bank.erase(image, target)

    assert paste is not None
    assert erase is not None
    _, paste_target, paste_mask = paste
    erase_image, erase_target, erase_mask = erase
    assert paste_target.sum() > 0
    assert paste_mask.sum() > 0
    assert erase_target.sum() == 0
    assert erase_mask.sum() > 0
    assert torch.isfinite(erase_image).all()


def test_lesion_intervention_bank_rejects_implausible_brain_placement() -> None:
    bank = LesionInterventionBank(
        max_patches=2,
        min_voxels=1,
        edge_softening=1,
        min_brain_coverage=1.0,
        placement_attempts=2,
    )
    image = torch.ones(1, 4, 12, 12, 12)
    target = torch.zeros(1, 3, 12, 12, 12)
    target[:, 1, 4:8, 4:8, 4:8] = 1.0
    bank.update(image, target)

    empty_recipient = torch.zeros_like(image)
    assert bank.paste(empty_recipient, torch.zeros_like(target)) is None


def test_nonenhancing_core_completion_adds_only_core_inside_wt() -> None:
    logits = torch.full((1, 3, 8, 8, 8), -8.0)
    logits[:, 1, 2:6, 2:6, 2:6] = 8.0
    batch = {
        "case_id": ["BT_TEST"],
        "metadata_raw": {
            "Tumor Type": ["ASTROCYTOMA, ANAPLASTIC"],
            "Tumor Grade": ["3.0"],
        },
    }

    completed, triggered = _apply_nonenhancing_core_completion(
        logits,
        batch,
        threshold=0.5,
        fraction=0.25,
        min_wt_voxels=16,
        max_tc_voxels=0,
        metadata_gate=True,
    )
    pred = torch.sigmoid(completed) >= 0.5

    assert triggered == [True]
    assert pred[:, 0].sum() > 0
    assert pred[:, 2].sum() == 0
    assert torch.all(pred[:, 0] <= pred[:, 1])


def test_nonenhancing_core_completion_metadata_gate_blocks_gbm() -> None:
    logits = torch.full((1, 3, 8, 8, 8), -8.0)
    logits[:, 1, 2:6, 2:6, 2:6] = 8.0
    batch = {
        "case_id": ["BT_TEST"],
        "metadata_raw": {
            "Tumor Type": ["GLIOBLASTOMA"],
            "Tumor Grade": ["4.0"],
        },
    }

    completed, triggered = _apply_nonenhancing_core_completion(
        logits,
        batch,
        threshold=0.5,
        fraction=0.25,
        min_wt_voxels=16,
        max_tc_voxels=0,
        metadata_gate=True,
    )

    assert triggered == [False]
    assert torch.allclose(completed, logits)


def test_component_consensus_demotion_only_changes_unstable_et_component() -> None:
    factual_prob = torch.full((1, 3, 8, 8, 8), 0.01)
    style_prob = factual_prob.clone()
    stable = (slice(1, 3), slice(1, 3), slice(1, 3))
    unstable = (slice(5, 7), slice(5, 7), slice(5, 7))
    factual_prob[:, 2, *stable] = 0.95
    factual_prob[:, 2, *unstable] = 0.95
    style_prob[:, 2, *stable] = 0.90
    style_prob[:, 2, *unstable] = 0.05

    fused = torch.sigmoid(
        _fuse_style_tta_logits(
            torch.logit(factual_prob.clamp(1e-4, 1.0 - 1e-4)),
            torch.logit(style_prob.clamp(1e-4, 1.0 - 1e-4)),
            "enhancing-component-consensus-demote-core",
        )
    )

    assert fused[:, 2, *stable].mean() > 0.80
    assert fused[:, 2, *unstable].mean() < 0.10
    assert fused[:, 0, *unstable].mean() > factual_prob[:, 0, *unstable].mean()


def test_phenotype_gated_demotion_uses_metadata_gate() -> None:
    factual_prob = torch.full((1, 3, 4, 4, 4), 0.01)
    style_prob = factual_prob.clone()
    factual_prob[:, 2, 1:3, 1:3, 1:3] = 0.95
    style_prob[:, 2, 1:3, 1:3, 1:3] = 0.05
    factual_logits = torch.logit(factual_prob.clamp(1e-4, 1.0 - 1e-4))
    style_logits = torch.logit(style_prob.clamp(1e-4, 1.0 - 1e-4))
    lower_grade_batch = {
        "metadata_raw": {
            "Tumor Type": ["ASTROCYTOMA, ANAPLASTIC"],
            "Tumor Grade": ["3.0"],
        }
    }
    gbm_batch = {
        "metadata_raw": {
            "Tumor Type": ["GLIOBLASTOMA"],
            "Tumor Grade": ["4.0"],
        }
    }

    lower_fused = torch.sigmoid(
        _fuse_style_tta_logits(
            factual_logits,
            style_logits,
            "phenotype-enhancing-demote-core",
            batch=lower_grade_batch,
        )
    )
    gbm_fused = torch.sigmoid(
        _fuse_style_tta_logits(
            factual_logits,
            style_logits,
            "phenotype-enhancing-demote-core",
            batch=gbm_batch,
        )
    )

    assert lower_fused[:, 2, 1:3, 1:3, 1:3].mean() < 0.10
    assert lower_fused[:, 0, 1:3, 1:3, 1:3].mean() > factual_prob[:, 0, 1:3, 1:3, 1:3].mean()
    assert torch.allclose(gbm_fused, factual_prob, atol=1e-4)
