import torch
from torch import nn

from crn.models import CounterfactualRepresentationNetwork


def test_unet_segmentation_head_produces_logits() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=12,
        num_classes=1,
        image_size=(64, 64),
        latent_dim=16,
        base_channels=8,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
    )
    batch = torch.randn(2, 12, 64, 64)
    outputs = model(batch)
    assert outputs["logits"].shape == (2, 1)
    assert outputs["seg_logits"].shape == (2, 3, 64, 64)
    assert outputs["seg_features"].shape[:2] == (2, 8)
    assert not model.supports_latent_segmentation
    seg_logits = model.segment_from_parts(outputs["z_d"], outputs["z_c"], outputs["disease_features"])
    assert seg_logits.shape == (2, 3, 64, 64)


def test_model_can_refresh_outputs_with_volume_context() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=1,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
    )
    batch = torch.randn(2, 4, 32, 32)
    outputs = model(batch)
    volume_context = torch.randn_like(outputs["z_c"])
    fused_context = model.fuse_context_latents(outputs["z_c_slice"], volume_context)
    refreshed = model.refresh_outputs(outputs, fused_context, volume_context=volume_context)
    assert refreshed["z_c"].shape == outputs["z_c"].shape
    assert refreshed["volume_context"].shape == outputs["z_c"].shape
    assert refreshed["seg_logits"].shape == (2, 3, 32, 32)
    assert refreshed["seg_features"].shape[1] == 4


def test_model_supports_group_norm_for_tiny_batch_training() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=0,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
        norm_type="group",
        group_norm_groups=4,
    )
    batch = torch.randn(1, 4, 32, 32)
    outputs = model(batch)
    assert outputs["seg_logits"].shape == (1, 3, 32, 32)
    assert any(isinstance(module, nn.GroupNorm) for module in model.modules())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())


def test_volumetric_backbone_produces_center_slice_segmentation() -> None:
    model = CounterfactualRepresentationNetwork(
        in_channels=4,
        num_classes=0,
        image_size=(32, 32),
        latent_dim=8,
        base_channels=4,
        num_seg_classes=3,
        head_uses_context=True,
        segmentation_head="unet",
        backbone_mode="2.5d",
        norm_type="group",
        group_norm_groups=4,
    )
    batch = torch.randn(2, 4, 5, 32, 32)
    outputs = model(batch)
    assert outputs["seg_logits"].shape == (2, 3, 32, 32)
    assert outputs["seg_features"].shape[:2] == (2, 4)
    assert outputs["reconstruction"].shape == (2, 4, 32, 32)
    seg_logits = model.segment_from_parts(outputs["z_d"], outputs["z_c"], outputs["disease_features"])
    assert seg_logits.shape == (2, 3, 32, 32)
