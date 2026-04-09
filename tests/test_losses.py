import torch

from crn.losses import bounded_stability_loss, decorrelation_loss


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

