import torch

from crn.losses import (
    CounterfactualMemory,
    LossWeights,
    backdoor_adjusted_seg_logits,
    brats_region_supervision_loss,
    brats_region_stability_loss,
    bounded_stability_loss,
    compute_crn_losses,
    counterfactual_contrastive_loss,
    decorrelation_loss,
)
from crn.models import CounterfactualRepresentationNetwork


def test_bounded_stability_allows_margin():
    original = torch.tensor([[0.10, 0.20]])
    counterfactual = torch.tensor([[0.15, 0.24]])
    loss = bounded_stability_loss(original, counterfactual, margin=0.10)
    assert loss.item() == 0.0


def test_bounded_stability_penalizes_excess():
    original = torch.tensor([[0.10, 0.80]])
    counterfactual = torch.tensor([[0.50, 0.20]])
    loss = bounded_stability_loss(original, counterfactual, margin=0.10)
    assert loss.item() > 0.0


def test_decorrelation_loss_is_scalar():
    z_d = torch.randn(4, 8)
    z_c = torch.randn(4, 8)
    loss = decorrelation_loss(z_d, z_c)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_decorrelation_loss_uses_reference_latents_for_single_sample() -> None:
    z_d = torch.tensor([[1.0, 0.0, 0.0]])
    z_c = torch.tensor([[1.0, 0.0, 0.0]])
    reference = (
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    loss = decorrelation_loss(z_d, z_c, reference_latents=reference)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_brats_region_supervision_loss_is_small_for_perfect_prediction():
    logits = torch.tensor(
        [
            [
                [[10.0, -10.0], [-10.0, -10.0]],
                [[-10.0, 10.0], [-10.0, -10.0]],
                [[-10.0, -10.0], [10.0, -10.0]],
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ]
    )
    loss = brats_region_supervision_loss(logits, target)
    assert loss.item() < 1e-2


def test_brats_region_stability_loss_respects_margin() -> None:
    original_logits = torch.tensor(
        [
            [
                [[10.0, -10.0], [-10.0, -10.0]],
                [[-10.0, 10.0], [-10.0, -10.0]],
                [[-10.0, -10.0], [10.0, -10.0]],
            ]
        ]
    )
    counterfactual_logits = original_logits - 0.1
    loss = brats_region_stability_loss(original_logits, counterfactual_logits, margin=0.05)
    assert loss.item() == 0.0


def test_counterfactual_contrastive_loss_prefers_positive_alignment() -> None:
    anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positive = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    hard_negative = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss = counterfactual_contrastive_loss(anchor, positive, hard_negative, temperature=0.1)
    assert torch.isfinite(loss)
    assert loss.item() < 0.2


def test_backdoor_adjusted_seg_logits_preserves_shape_for_unet() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=0,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
    )
    batch = torch.randn(2, 4, 32, 32)
    outputs = model(batch)
    adjusted = backdoor_adjusted_seg_logits(
        model,
        outputs["disease_features"],
        outputs["z_d"],
        outputs["z_c"],
        max_contexts=2,
    )
    assert adjusted.shape == (2, 3, 32, 32)
    assert torch.isfinite(adjusted).all()


def test_counterfactual_memory_provides_volume_context_bank() -> None:
    memory = CounterfactualMemory(
        num_samples=6,
        volume_ids=[10, 20, 30],
        latent_dim=8,
        device=torch.device("cpu"),
        match_topk=2,
        exemplar_capacity=4,
    )
    sample_indices = torch.tensor([0, 1, 2])
    volume_ids = torch.tensor([10, 20, 30])
    z_d = torch.randn(3, 8)
    z_c = torch.randn(3, 8)
    mask = (torch.rand(3, 3, 8, 8) > 0.7).float()
    labels = torch.rand(3, 1)
    disease_features = (
        torch.randn(3, 4, 8, 8),
        torch.randn(3, 8, 4, 4),
    )
    memory.update(sample_indices, volume_ids, z_d, z_c)
    memory.store_exemplars(sample_indices, volume_ids, z_d, z_c, mask=mask, label=labels, disease_features=disease_features)
    bank = memory.context_bank(max_contexts=2, device=torch.device("cpu"))
    lookup = memory.lookup_volume_context(volume_ids, torch.zeros_like(z_c))
    matches = memory.matched_contexts(z_d, z_c, sample_indices, volume_ids)
    reference = memory.reference_latents(sample_indices=sample_indices[:1], device=torch.device("cpu"))
    disease_examples = memory.matched_disease_examples(
        z_d[:1],
        z_c[:1],
        sample_indices[:1],
        volume_ids[:1],
        require_mask=True,
        require_label=True,
        require_features=True,
    )
    assert bank is not None
    assert bank.shape == (2, 8)
    assert lookup.shape == (3, 8)
    assert matches is not None
    assert matches.shape == (3, 8)
    assert reference is not None
    assert reference[0].shape[0] == 2
    assert disease_examples is not None
    assert disease_examples["z_d"].shape == (1, 8)
    assert disease_examples["mask"].shape == (1, 3, 8, 8)
    assert disease_examples["label"].shape == (1, 1)
    assert len(disease_examples["disease_features"]) == 2
    assert torch.isfinite(disease_examples["z_d"]).all()
    assert torch.isfinite(disease_examples["mask"]).all()
    assert disease_examples["mask"].min().item() >= 0.0
    assert disease_examples["mask"].max().item() <= 1.0


def test_compute_crn_losses_supports_causal_segmentation_terms_for_unet() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=0,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
    )
    image = torch.randn(2, 4, 32, 32)
    mask = (torch.rand(2, 3, 32, 32) > 0.8).float()
    outputs = model(image)
    total, logs = compute_crn_losses(
        model,
        {"image": image, "mask": mask},
        outputs,
        LossWeights(
            lambda_seg=1.0,
            lambda_region_adjustment=0.25,
            lambda_region_cf_stability=0.05,
            lambda_region_disease_swap=0.10,
            adjustment_contexts=2,
        ),
    )
    assert torch.isfinite(total)
    assert "loss/region_adjustment" in logs
    assert "loss/region_cf_stability" in logs
    assert "loss/region_disease_swap" in logs


def test_compute_crn_losses_supports_region_counterfactual_contrastive_loss() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=0,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
    )
    image = torch.randn(2, 4, 32, 32)
    mask = torch.zeros(2, 3, 32, 32)
    mask[0, 0, 4:14, 4:14] = 1.0
    mask[1, 2, 12:24, 12:24] = 1.0
    outputs = model(image)
    total, logs = compute_crn_losses(
        model,
        {"image": image, "mask": mask},
        outputs,
        LossWeights(
            lambda_seg=1.0,
            lambda_region_cf_contrastive=0.15,
            adjustment_contexts=2,
            contrastive_temperature=0.2,
        ),
    )
    assert torch.isfinite(total)
    assert "loss/region_cf_contrastive" in logs
    assert logs["loss/region_cf_contrastive"].item() >= 0.0


def test_compute_crn_losses_uses_counterfactual_memory_when_available() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=0,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
    )
    image = torch.randn(3, 4, 32, 32)
    mask = (torch.rand(3, 3, 32, 32) > 0.8).float()
    outputs = model(image)
    memory = CounterfactualMemory(
        num_samples=8,
        volume_ids=[0, 1, 2],
        latent_dim=8,
        device=torch.device("cpu"),
        match_topk=2,
    )
    batch = {
        "image": image,
        "mask": mask,
        "index": torch.tensor([0, 1, 2]),
        "volume": torch.tensor([0, 1, 2]),
    }
    memory.update(batch["index"], batch["volume"], outputs["z_d"], outputs["z_c_slice"])
    outputs = model.refresh_outputs(outputs, model.fuse_context_latents(outputs["z_c_slice"], memory.lookup_volume_context(batch["volume"], outputs["z_c_slice"])))
    total, logs = compute_crn_losses(
        model,
        batch,
        outputs,
        LossWeights(
            lambda_seg=1.0,
            lambda_region_adjustment=0.25,
            lambda_region_cf_stability=0.05,
            lambda_region_disease_swap=0.10,
            adjustment_contexts=2,
        ),
        counterfactual_memory=memory,
    )
    assert torch.isfinite(total)
    assert logs["loss/region_adjustment"].item() >= 0.0


def test_compute_crn_losses_uses_memory_for_single_sample_causal_terms() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=0,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
    )
    memory = CounterfactualMemory(
        num_samples=8,
        volume_ids=[0, 1, 2],
        latent_dim=8,
        device=torch.device("cpu"),
        match_topk=2,
        exemplar_capacity=4,
    )

    donor_image = torch.randn(2, 4, 32, 32)
    donor_mask = torch.zeros(2, 3, 32, 32)
    donor_mask[0, 0, 4:12, 4:12] = 1.0
    donor_mask[1, 2, 12:20, 12:20] = 1.0
    donor_outputs = model(donor_image)
    donor_batch = {
        "image": donor_image,
        "mask": donor_mask,
        "index": torch.tensor([0, 1]),
        "volume": torch.tensor([0, 1]),
    }
    memory.update(donor_batch["index"], donor_batch["volume"], donor_outputs["z_d"], donor_outputs["z_c_slice"])
    donor_outputs = model.refresh_outputs(
        donor_outputs,
        model.fuse_context_latents(
            donor_outputs["z_c_slice"],
            memory.lookup_volume_context(donor_batch["volume"], donor_outputs["z_c_slice"]),
        ),
    )
    memory.store_exemplars(
        donor_batch["index"],
        donor_batch["volume"],
        donor_outputs["z_d"],
        donor_outputs["z_c"],
        mask=donor_batch["mask"],
        disease_features=donor_outputs["disease_features"],
    )

    image = torch.randn(1, 4, 32, 32)
    mask = torch.zeros(1, 3, 32, 32)
    mask[0, 1, 8:16, 8:16] = 1.0
    batch = {
        "image": image,
        "mask": mask,
        "index": torch.tensor([2]),
        "volume": torch.tensor([2]),
    }
    outputs = model(image)
    outputs = model.refresh_outputs(
        outputs,
        model.fuse_context_latents(
            outputs["z_c_slice"],
            memory.lookup_volume_context(batch["volume"], outputs["z_c_slice"]),
        ),
    )
    total, logs = compute_crn_losses(
        model,
        batch,
        outputs,
        LossWeights(
            lambda_seg=1.0,
            lambda_dis=0.1,
            lambda_region_adjustment=0.25,
            lambda_region_disease_swap=0.10,
            adjustment_contexts=2,
        ),
        counterfactual_memory=memory,
    )
    assert torch.isfinite(total)
    assert logs["loss/dis"].item() > 0.0
    assert logs["loss/region_disease_swap"].item() > 0.0


def test_compute_crn_losses_supports_volumetric_backbone() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=0,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
        backbone_mode="3d",
        norm_type="group",
        group_norm_groups=4,
    )
    image = torch.randn(2, 4, 5, 32, 32)
    mask = (torch.rand(2, 3, 32, 32) > 0.8).float()
    outputs = model(image)
    total, logs = compute_crn_losses(
        model,
        {"image": image, "mask": mask},
        outputs,
        LossWeights(
            lambda_seg=1.0,
            lambda_region_adjustment=0.25,
            lambda_region_cf_stability=0.05,
            lambda_region_disease_swap=0.10,
            lambda_region_cf_contrastive=0.08,
            adjustment_contexts=2,
        ),
    )
    assert torch.isfinite(total)
    assert "loss/region_adjustment" in logs
    assert "loss/region_disease_swap" in logs
    assert "loss/region_cf_contrastive" in logs
