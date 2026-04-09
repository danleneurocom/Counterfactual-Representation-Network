import torch

from crn.metrics import binary_segmentation_metrics, brats_region_metrics, classification_metrics


def test_binary_segmentation_metrics_returns_dice():
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    metrics = binary_segmentation_metrics(logits, target)
    assert metrics["seg/slice_dice_mean"] > 0.99


def test_classification_metrics_binary():
    logits = torch.tensor([[5.0], [-5.0], [4.0], [-4.0]])
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    metrics = classification_metrics(logits, target)
    assert metrics["cls/accuracy"] == 1.0
    assert metrics["cls/auroc"] == 1.0


def test_brats_region_metrics_returns_region_scores():
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
    metrics = brats_region_metrics(logits, target)
    assert metrics["brats/WT/dice"] > 0.99
    assert metrics["brats/TC/dice"] > 0.99
    assert metrics["brats/ET/dice"] > 0.99
