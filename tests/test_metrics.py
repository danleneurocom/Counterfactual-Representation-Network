import math

import pytest
import torch

from crn.metrics import (
    binary_segmentation_metrics,
    binary_volume_metrics_from_masks,
    brats_region_metrics,
    brats_volume_metrics_from_probs,
    classification_metrics,
    postprocess_binary_volume,
)


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


def test_brats_volume_metrics_returns_region_scores():
    probs = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ]
    )
    target = probs.clone()
    metrics = brats_volume_metrics_from_probs(probs, target)
    assert metrics["brats/WT/dice"] > 0.99
    assert metrics["brats/TC/dice"] > 0.99
    assert metrics["brats/ET/dice"] > 0.99
    assert metrics["brats/WT/hd95"] == pytest.approx(0.0)
    assert metrics["brats/TC/hd95"] == pytest.approx(0.0)
    assert metrics["brats/ET/hd95"] == pytest.approx(0.0)


def test_brats_volume_metrics_penalize_empty_prediction_hd95():
    probs = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    target = torch.zeros_like(probs)
    target[0, 2, 0, 0] = 1.0

    metrics = brats_volume_metrics_from_probs(probs, target)

    expected_diagonal = math.sqrt(3.0)
    assert metrics["brats/ET/hd95"] == pytest.approx(expected_diagonal)
    assert metrics["brats/WT/hd95"] == pytest.approx(expected_diagonal)


def test_postprocess_binary_volume_fills_holes_and_removes_small_components():
    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[1:4, 1:4] = True
    mask[2, 2] = False
    mask[0, 0] = True

    processed = postprocess_binary_volume(mask.numpy(), min_component_size=4, fill_holes=True)

    assert processed[2, 2]
    assert not processed[0, 0]


def test_binary_volume_metrics_from_masks_match_has_zero_hd95():
    pred = torch.zeros((2, 3, 3), dtype=torch.bool)
    pred[0, 1, 1] = True
    pred[1, 1, 1] = True
    target = pred.clone()

    metrics = binary_volume_metrics_from_masks(pred.numpy(), target.numpy())

    assert metrics["dice"] > 0.99
    assert metrics["hd95"] == pytest.approx(0.0)
