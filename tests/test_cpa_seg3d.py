import torch

from cpa_seg3d import build_cpa_seg3d_tiny, boundary_targets_from_subregions, region_targets_from_subregions


def test_cpa_seg3d_tiny_forward_exposes_region_boundary_and_adjustment_outputs() -> None:
    torch.manual_seed(7)
    model = build_cpa_seg3d_tiny(
        latent_dim=16,
        context_proxy_dim=3,
        disease_proxy_dim=2,
        annotation_proxy_dim=1,
    )
    model.eval()

    image = torch.randn(1, 4, 32, 32, 32)
    context_bank = torch.randn(3, 16)
    with torch.no_grad():
        outputs = model(image, context_bank=context_bank, max_adjustment_contexts=2)

    assert outputs["logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["region_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["boundary_logits"].shape == (1, 1, 32, 32, 32)
    assert outputs["adjusted_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["adjusted_region_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["context_proxy_logits"].shape == (1, 3)
    assert outputs["disease_proxy_logits"].shape == (1, 2)
    assert outputs["annotation_proxy_logits"].shape == (1, 1)
    assert len(outputs["deep_logits"]) == 3


def test_cpa_seg3d_unet_decoder_variant_still_runs() -> None:
    model = build_cpa_seg3d_tiny(latent_dim=16, decoder_variant="unet")
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 4, 32, 32, 32))

    assert outputs["logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["region_logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["boundary_logits"].shape == (1, 1, 32, 32, 32)


def test_region_and_boundary_targets_from_subregions() -> None:
    mask = torch.zeros(1, 3, 8, 8, 8)
    mask[:, 0, 2:5, 2:5, 2:5] = 1.0
    mask[:, 1, 1:7, 1:7, 1:7] = 1.0
    mask[:, 2, 3:4, 3:4, 3:4] = 1.0

    regions = region_targets_from_subregions(mask)
    boundary = boundary_targets_from_subregions(mask)

    assert regions.shape == (1, 3, 8, 8, 8)
    assert boundary.shape == (1, 1, 8, 8, 8)
    assert torch.equal(regions[:, 2], mask[:, 2])
    assert regions[:, 0].sum() >= regions[:, 1].sum() >= regions[:, 2].sum()
    assert boundary.max() == 1.0
    assert boundary.sum() > 0
