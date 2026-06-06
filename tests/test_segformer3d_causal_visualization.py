from __future__ import annotations

import torch

from baselines.segformer3d.visualize_causal_utsw import (
    region_binary_mask,
    region_probability_map,
    select_slice_index,
)


def test_region_probability_map_uses_noisy_or_for_union_regions() -> None:
    probs = torch.zeros(3, 2, 2, 2)
    probs[0, 0, 0, 0] = 0.20
    probs[1, 0, 0, 0] = 0.50
    probs[2, 0, 0, 0] = 0.25

    wt = region_probability_map(probs, "WT")
    tc = region_probability_map(probs, "TC")
    et = region_probability_map(probs, "ET")

    assert torch.isclose(wt[0, 0, 0], torch.tensor(1.0 - 0.80 * 0.50 * 0.75))
    assert torch.isclose(tc[0, 0, 0], torch.tensor(1.0 - 0.80 * 0.75))
    assert torch.isclose(et[0, 0, 0], torch.tensor(0.25))


def test_region_binary_mask_uses_union_for_brats_regions() -> None:
    mask = torch.zeros(3, 3, 2, 2)
    mask[0, 1, 0, 0] = 1.0
    mask[1, 2, 0, 0] = 1.0
    mask[2, 0, 0, 0] = 1.0

    wt = region_binary_mask(mask, "WT")
    tc = region_binary_mask(mask, "TC")
    et = region_binary_mask(mask, "ET")

    assert wt[:, 0, 0].tolist() == [1.0, 1.0, 1.0]
    assert tc[:, 0, 0].tolist() == [1.0, 1.0, 0.0]
    assert et[:, 0, 0].tolist() == [1.0, 0.0, 0.0]


def test_select_slice_prefers_ground_truth_then_prediction_then_effect() -> None:
    target = torch.zeros(4, 2, 2)
    factual = torch.zeros(4, 2, 2)
    adjusted = torch.zeros(4, 2, 2)
    effect = torch.zeros(4, 2, 2)
    target[2] = 1.0
    factual[1] = 0.9
    effect[3] = 0.5

    assert select_slice_index(target, factual, adjusted, effect, threshold=0.5) == 2

    target.zero_()
    assert select_slice_index(target, factual, adjusted, effect, threshold=0.5) == 1

    factual.zero_()
    assert select_slice_index(target, factual, adjusted, effect, threshold=0.5) == 3

    effect.zero_()
    assert select_slice_index(target, factual, adjusted, effect, threshold=0.5) == 2
