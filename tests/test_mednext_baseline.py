from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import pandas as pd
import torch

from baselines.mednext.common import (
    checkpoint_monitor,
    initialize_model_from_checkpoint,
    initialize_output_bias_from_loader,
    main_logits,
    registered_modality_consistency_metrics,
    segmentation_loss,
)
from baselines.mednext.causal import CausalMedNeXt, build_causal_mednext
from baselines.mednext.calibration import (
    BratsAdaptiveRegionThresholdSweep,
    BratsRegionThresholdSweep,
    brats_region_metrics_from_thresholds,
    brats_region_metrics_from_plausibility_thresholds,
    fit_plausibility_support_thresholds,
    parse_fraction_candidates,
    parse_region_thresholds,
)
from baselines.mednext.dataset_cache import DiskCachedDataset, maybe_disk_cache_dataset, warm_disk_cache_dataset
from baselines.mednext import evaluate_causal_brats_h5 as mednext_causal_brats_eval
from baselines.mednext.evaluate_causal_brats_h5 import _should_build_context_bank as _should_build_brats_context_bank
from baselines.mednext.evaluate_causal_utsw import _should_build_context_bank as _should_build_utsw_context_bank
from baselines.mednext.evaluate_causal_utsw import (
    _apply_nonenhancing_core_completion,
    _case_record,
    _fuse_registered_modality_logits,
    _fuse_style_tta_logits,
    _registered_modality_stability_gate_mask,
)
from baselines.mednext.model import build_mednext_segmenter
from baselines.mednext.roi_refiner import CausalRoiRefiner, bbox_from_mask, crop_resize_3d, paste_resized_3d, scale_bbox
from baselines.mednext.train_brats_h5 import _cache_signature as _brats_baseline_cache_signature
from baselines.mednext.train_causal_brats_h5 import _cache_signature as _brats_causal_cache_signature
from baselines.mednext import train_causal_brats_h5 as mednext_causal_brats_train
from baselines.mednext import train_causal_utsw as mednext_causal_train
from baselines.mednext.train_causal_utsw import (
    LesionInterventionBank,
    _et_volume_veto_scale_for_epoch,
    _load_causal_init_checkpoint,
    apply_style_intervention,
)
from crn.metrics import brats_region_metrics
from baselines.segformer3d.train_causal_utsw import (
    _balanced_region_mediator_loss,
    _balanced_subregion_mediator_loss,
    _boundary_mediator_loss,
    _frontdoor_router_advantage_loss,
    _prototype_mediator_loss,
    _subregion_class_target,
)


def test_utsw_causal_parse_args_loads_config_json_and_cli_overrides(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "baseline_checkpoint": "saved-baseline.pt",
                "output_dir": "saved-output",
                "volume_size": 48,
                "base_channels": 8,
                "checkpoint_registered_modality_selector": "val-loss",
                "lambda_registered_modality_region_consistency": 0.0,
                "lambda_registered_modality_wt_consistency": 0.0,
                "registered_modality_channel_loss_weights": "0.5,1.0,2.5",
                "registered_modality_region_loss_weights": "0.2,2.0,3.0",
                "registered_modality_weight_ramp_epochs": 3,
                "registered_modality_small_lesion_emphasis": 0.25,
                "registered_modality_small_lesion_reference_fractions": "0.02,0.01,0.005",
                "registered_modality_small_lesion_region_weights": "0.5,1.0,2.0",
                "registered_modality_small_lesion_max_weight": 1.5,
                "registered_modality_error_emphasis": 0.2,
                "registered_modality_error_region_weights": "0.25,1.0,2.0",
                "registered_modality_error_max_weight": 1.4,
                "lambda_registered_modality_fusion_seg": 0.1,
                "registered_modality_fusion_mode": "max-probs",
                "lambda_registered_modality_view_advantage_distillation": 0.05,
                "registered_modality_view_advantage_region_weights": "0.5,1.0,2.0",
                "registered_modality_view_advantage_margin": 0.03,
                "hard_case_sampler_emphasis": 0.0,
                "hard_case_sampler_reference_fractions": "0.02,0.01,0.005",
                "hard_case_sampler_region_weights": "0.5,1.0,2.0",
                "hard_case_sampler_max_weight": 2.5,
                "hard_case_sampler_epoch_multiplier": 1.25,
                "unknown_future_key": "ignored",
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_causal_utsw.py",
            "--config-json",
            str(config_path),
            "--output-dir",
            "override-output",
            "--lambda-registered-modality-region-consistency",
            "0.15",
            "--lambda-registered-modality-wt-consistency",
            "0.3",
            "--registered-modality-small-lesion-emphasis",
            "0.75",
            "--registered-modality-error-emphasis",
            "0.6",
            "--lambda-registered-modality-fusion-seg",
            "0.3",
            "--lambda-registered-modality-view-advantage-distillation",
            "0.2",
            "--hard-case-sampler-emphasis",
            "1.5",
        ],
    )

    args = mednext_causal_train.parse_args()

    assert args.baseline_checkpoint == "saved-baseline.pt"
    assert args.output_dir == "override-output"
    assert args.volume_size == 48
    assert args.base_channels == 8
    assert args.checkpoint_registered_modality_selector == "val-loss"
    assert args.lambda_registered_modality_region_consistency == 0.15
    assert args.lambda_registered_modality_wt_consistency == 0.3
    assert args.registered_modality_channel_loss_weights == "0.5,1.0,2.5"
    assert args.registered_modality_region_loss_weights == "0.2,2.0,3.0"
    assert args.registered_modality_weight_ramp_epochs == 3
    assert args.registered_modality_small_lesion_emphasis == 0.75
    assert args.registered_modality_small_lesion_reference_fractions == "0.02,0.01,0.005"
    assert args.registered_modality_small_lesion_region_weights == "0.5,1.0,2.0"
    assert args.registered_modality_small_lesion_max_weight == 1.5
    assert args.registered_modality_error_emphasis == 0.6
    assert args.registered_modality_error_region_weights == "0.25,1.0,2.0"
    assert args.registered_modality_error_max_weight == 1.4
    assert args.lambda_registered_modality_fusion_seg == 0.3
    assert args.registered_modality_fusion_mode == "max-probs"
    assert args.lambda_registered_modality_view_advantage_distillation == 0.2
    assert args.registered_modality_view_advantage_region_weights == "0.5,1.0,2.0"
    assert args.registered_modality_view_advantage_margin == 0.03
    assert args.hard_case_sampler_emphasis == 1.5
    assert args.hard_case_sampler_reference_fractions == "0.02,0.01,0.005"
    assert args.hard_case_sampler_region_weights == "0.5,1.0,2.0"
    assert args.hard_case_sampler_max_weight == 2.5
    assert args.hard_case_sampler_epoch_multiplier == 1.25


def test_brats_causal_parse_args_loads_config_json_and_cli_overrides(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "brats_config.json"
    config_path.write_text(
        json.dumps(
            {
                "baseline_checkpoint": "saved-brats-baseline.pt",
                "output_dir": "saved-brats-output",
                "train_csv": "saved-train.csv",
                "val_csv": "saved-val.csv",
                "limit_train_volumes": 64,
                "max_train_batches": 64,
                "max_context_bank_batches": 64,
                "hard_case_sampler_emphasis": 0.75,
                "hard_case_sampler_reference_fractions": "0.015,0.006,0.003",
            "hard_case_sampler_region_weights": "0.25,1.0,2.5",
            "hard_case_sampler_max_weight": 2.0,
            "warm_disk_cache_split": "train",
            "unknown_future_key": "ignored",
        }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_causal_brats_h5.py",
            "--config-json",
            str(config_path),
            "--output-dir",
            "override-brats-output",
            "--epochs",
            "1",
            "--hard-case-sampler-emphasis",
            "0.5",
            "--warm-disk-cache-only",
            "--warm-disk-cache-max-items",
            "2",
        ],
    )

    args = mednext_causal_brats_train.parse_args()

    assert args.baseline_checkpoint == "saved-brats-baseline.pt"
    assert args.output_dir == "override-brats-output"
    assert args.train_csv == "saved-train.csv"
    assert args.val_csv == "saved-val.csv"
    assert args.epochs == 1
    assert args.limit_train_volumes == 64
    assert args.max_train_batches == 64
    assert args.max_context_bank_batches == 64
    assert args.hard_case_sampler_emphasis == 0.5
    assert args.hard_case_sampler_reference_fractions == "0.015,0.006,0.003"
    assert args.hard_case_sampler_region_weights == "0.25,1.0,2.5"
    assert args.hard_case_sampler_max_weight == 2.0
    assert args.warm_disk_cache_only is True
    assert args.warm_disk_cache_split == "train"
    assert args.warm_disk_cache_max_items == 2


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


def test_mednext_balanced_focal_loss_has_finite_gradients_on_sparse_masks() -> None:
    logits = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
    target = torch.zeros(1, 3, 8, 8, 8)
    target[:, 2, 3:5, 3:5, 3:5] = 1.0
    args = type(
        "Args",
        (),
        {
            "seg_loss_mode": "balanced_focal",
            "channel_loss_weights": "1.0,1.0,2.0",
            "region_loss_weights": "1.0,1.5,2.5",
            "lambda_region_loss": 0.1,
            "balanced_bce_max_pos_weight": 20.0,
            "focal_tversky_alpha": 0.6,
            "focal_tversky_beta": 0.4,
            "focal_tversky_gamma": 0.75,
            "lambda_volume_prior_loss": 0.05,
            "volume_prior_scale": 1000.0,
        },
    )()

    loss = segmentation_loss(logits, target, args=args)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_mednext_causal_loss_dispatch_uses_mednext_segmentation_terms(monkeypatch) -> None:
    calls: list[str] = []

    def fake_segmentation_terms(logits: torch.Tensor, target: torch.Tensor, args: object, prefix: str) -> dict[str, torch.Tensor]:
        calls.append(prefix)
        return {prefix: logits.sum() * 0.0 + 1.0}

    monkeypatch.setattr(mednext_causal_train, "_segmentation_terms", fake_segmentation_terms)
    args = type("Args", (), {"proxy_loss_mode": "typed"})()
    outputs = {
        "logits": torch.zeros(1, 3, 4, 4, 4),
        "z_d": torch.ones(1, 4),
        "z_c": torch.zeros(1, 4),
    }
    batch = {"mask": torch.zeros(1, 3, 4, 4, 4)}

    terms = mednext_causal_train._causal_loss_terms(outputs, batch, args, proxy_layout=None)

    assert calls == ["seg"]
    assert torch.allclose(terms["seg"], torch.tensor(1.0))


def test_registered_modality_pair_dataset_aligns_cases_and_delegates_metadata() -> None:
    class TinyCaseDataset(torch.utils.data.Dataset):
        metadata_encoder = "metadata"

        def __init__(self, offset: float) -> None:
            self.offset = float(offset)
            self.case_ids = ["BT_A", "BT_B"]

        def __len__(self) -> int:
            return len(self.case_ids)

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "case_id": self.case_ids[index],
                "image": torch.full((1,), self.offset + index),
                "mask": torch.zeros(1),
            }

    paired = mednext_causal_train.RegisteredModalityPairDataset(
        TinyCaseDataset(0.0),
        TinyCaseDataset(10.0),
    )

    item = paired[1]

    assert len(paired) == 2
    assert paired.metadata_encoder == "metadata"
    assert item["case_id"] == "BT_B"
    assert item["registered_case_id"] == "BT_B"
    assert torch.equal(item["image"], torch.tensor([1.0]))
    assert torch.equal(item["registered_image"], torch.tensor([11.0]))


def test_registered_modality_terms_and_weights_are_active() -> None:
    args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 0.25,
            "lambda_registered_modality_consistency": 0.5,
            "lambda_registered_modality_disease_invariance": 0.75,
            "lambda_region_loss": 0.1,
            "seg_loss_mode": "bce_dice",
            "channel_loss_weights": "1.0,1.0,1.0",
            "region_loss_weights": "1.0,1.0,1.0",
            "lambda_volume_prior_loss": 0.0,
            "distill_channel_weights": "1.0,1.0,1.0",
        },
    )()
    outputs = {
        "logits": torch.zeros(1, 3, 4, 4, 4),
        "z_d": torch.tensor([[1.0, 0.0]]),
    }
    registered_outputs = {
        "logits": torch.ones(1, 3, 4, 4, 4) * 0.25,
        "z_d": torch.tensor([[0.0, 1.0]]),
    }
    target = torch.zeros(1, 3, 4, 4, 4)
    target[:, 2, 1:3, 1:3, 1:3] = 1.0

    terms = mednext_causal_train._registered_modality_terms(outputs, registered_outputs, target, args)

    assert {
        "registered_modality_seg",
        "registered_modality_seg_region",
        "registered_modality_consistency",
        "registered_modality_disease_invariance",
    }.issubset(terms)
    assert all(torch.isfinite(value) for value in terms.values())

    synthetic_terms = {
        "registered_modality_seg": torch.tensor(2.0),
        "registered_modality_seg_region": torch.tensor(3.0),
        "registered_modality_consistency": torch.tensor(5.0),
        "registered_modality_disease_invariance": torch.tensor(7.0),
    }
    total = mednext_causal_train._weighted_total(synthetic_terms, args)

    assert torch.allclose(total, torch.tensor(9.0))


def test_registered_modality_seg_uses_registered_specific_weights() -> None:
    args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 1.0,
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 0.0,
            "lambda_registered_modality_wt_consistency": 0.0,
            "lambda_registered_modality_disease_invariance": 0.0,
            "lambda_region_loss": 1.0,
            "seg_loss_mode": "bce_dice",
            "channel_loss_weights": "1.0,1.0,1.0",
            "region_loss_weights": "1.0,1.0,1.0",
            "registered_modality_channel_loss_weights": "0.2,1.0,5.0",
            "registered_modality_region_loss_weights": "0.2,1.0,5.0",
            "lambda_volume_prior_loss": 0.0,
            "distill_channel_weights": "1.0,1.0,1.0",
        },
    )()
    uniform_args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 1.0,
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 0.0,
            "lambda_registered_modality_wt_consistency": 0.0,
            "lambda_registered_modality_disease_invariance": 0.0,
            "lambda_region_loss": 1.0,
            "seg_loss_mode": "bce_dice",
            "channel_loss_weights": "1.0,1.0,1.0",
            "region_loss_weights": "1.0,1.0,1.0",
            "registered_modality_channel_loss_weights": "",
            "registered_modality_region_loss_weights": "",
            "lambda_volume_prior_loss": 0.0,
            "distill_channel_weights": "1.0,1.0,1.0",
        },
    )()
    registered_logits = torch.zeros(1, 3, 4, 4, 4)
    registered_logits[:, 0] = 3.0
    registered_logits[:, 2] = -3.0
    target = torch.zeros_like(registered_logits)
    target[:, 0, :2] = 1.0
    target[:, 2, 3:] = 1.0

    weighted = mednext_causal_train._registered_modality_terms(
        {"logits": torch.zeros_like(registered_logits)},
        {"logits": registered_logits},
        target,
        args,
    )
    uniform = mednext_causal_train._registered_modality_terms(
        {"logits": torch.zeros_like(registered_logits)},
        {"logits": registered_logits},
        target,
        uniform_args,
    )

    assert weighted["registered_modality_seg"] > uniform["registered_modality_seg"]
    assert weighted["registered_modality_seg_region"] > uniform["registered_modality_seg_region"]
    assert args.channel_loss_weights == "1.0,1.0,1.0"
    assert args.region_loss_weights == "1.0,1.0,1.0"


def test_registered_modality_weight_schedule_interpolates_from_base_weights() -> None:
    args = type(
        "Args",
        (),
        {
            "channel_loss_weights": "1.0,1.2,2.0",
            "region_loss_weights": "1.0,1.5,2.5",
            "registered_modality_channel_loss_weights": "0.5,1.0,2.5",
            "registered_modality_region_loss_weights": "0.2,2.0,3.0",
            "registered_modality_start_channel_loss_weights": "",
            "registered_modality_start_region_loss_weights": "",
            "registered_modality_weight_ramp_epochs": 3,
        },
    )()

    setattr(args, "_current_epoch", 1)
    epoch1 = mednext_causal_train._registered_segmentation_args(args)
    setattr(args, "_current_epoch", 2)
    epoch2 = mednext_causal_train._registered_segmentation_args(args)
    setattr(args, "_current_epoch", 3)
    epoch3 = mednext_causal_train._registered_segmentation_args(args)

    def weights(spec: str) -> list[float]:
        return [float(item) for item in spec.split(",")]

    assert weights(epoch1.channel_loss_weights) == [1.0, 1.2, 2.0]
    assert weights(epoch1.region_loss_weights) == [1.0, 1.5, 2.5]
    assert weights(epoch2.channel_loss_weights) == [0.75, 1.1, 2.25]
    assert weights(epoch2.region_loss_weights) == [0.6, 1.75, 2.75]
    assert weights(epoch3.channel_loss_weights) == [0.5, 1.0, 2.5]
    assert weights(epoch3.region_loss_weights) == [0.2, 2.0, 3.0]
    assert args.channel_loss_weights == "1.0,1.2,2.0"
    assert args.region_loss_weights == "1.0,1.5,2.5"


def test_registered_modality_small_lesion_weight_boosts_only_nonempty_small_targets() -> None:
    args = type(
        "Args",
        (),
        {
            "registered_modality_small_lesion_emphasis": 1.0,
            "registered_modality_small_lesion_reference_fractions": "0.2,0.1,0.05",
            "registered_modality_small_lesion_region_weights": "0.5,1.0,2.0",
            "registered_modality_small_lesion_max_weight": 1.75,
        },
    )()
    small_target = torch.zeros(1, 3, 10, 10, 10)
    small_target[:, 2, 0, 0, 0] = 1.0
    large_target = torch.zeros_like(small_target)
    large_target[:, 0, :6] = 1.0
    large_target[:, 2, :6] = 1.0
    empty_target = torch.zeros_like(small_target)

    small_weight = mednext_causal_train._registered_modality_small_lesion_weight(small_target, args)
    large_weight = mednext_causal_train._registered_modality_small_lesion_weight(large_target, args)
    empty_weight = mednext_causal_train._registered_modality_small_lesion_weight(empty_target, args)

    assert small_weight > large_weight
    assert small_weight <= torch.tensor(1.75)
    assert torch.allclose(large_weight, torch.tensor(1.0))
    assert torch.allclose(empty_weight, torch.tensor(1.0))


def test_registered_modality_seg_applies_small_lesion_weight() -> None:
    common = {
        "lambda_registered_modality_seg": 1.0,
        "lambda_registered_modality_consistency": 0.0,
        "lambda_registered_modality_region_consistency": 0.0,
        "lambda_registered_modality_wt_consistency": 0.0,
        "lambda_registered_modality_disease_invariance": 0.0,
        "lambda_region_loss": 1.0,
        "seg_loss_mode": "bce_dice",
        "channel_loss_weights": "1.0,1.0,1.0",
        "region_loss_weights": "1.0,1.0,1.0",
        "registered_modality_channel_loss_weights": "",
        "registered_modality_region_loss_weights": "",
        "lambda_volume_prior_loss": 0.0,
        "distill_channel_weights": "1.0,1.0,1.0",
        "registered_modality_small_lesion_reference_fractions": "0.2,0.1,0.05",
        "registered_modality_small_lesion_region_weights": "0.5,1.0,2.0",
        "registered_modality_small_lesion_max_weight": 1.75,
    }
    boosted_args = type("Args", (), {**common, "registered_modality_small_lesion_emphasis": 1.0})()
    base_args = type("Args", (), {**common, "registered_modality_small_lesion_emphasis": 0.0})()
    registered_logits = torch.zeros(1, 3, 10, 10, 10)
    target = torch.zeros_like(registered_logits)
    target[:, 2, 0, 0, 0] = 1.0

    boosted = mednext_causal_train._registered_modality_terms(
        {"logits": torch.zeros_like(registered_logits)},
        {"logits": registered_logits},
        target,
        boosted_args,
    )
    base = mednext_causal_train._registered_modality_terms(
        {"logits": torch.zeros_like(registered_logits)},
        {"logits": registered_logits},
        target,
        base_args,
    )

    assert boosted["registered_modality_seg"] > base["registered_modality_seg"]
    assert boosted["registered_modality_seg_region"] > base["registered_modality_seg_region"]


def test_registered_modality_error_weight_tracks_factual_region_error() -> None:
    args = type(
        "Args",
        (),
        {
            "registered_modality_error_emphasis": 1.0,
            "registered_modality_error_region_weights": "0.25,1.0,2.0",
            "registered_modality_error_max_weight": 2.0,
        },
    )()
    target = torch.zeros(1, 3, 6, 6, 6)
    target[:, 2, 2:4, 2:4, 2:4] = 1.0
    good_logits = torch.full_like(target, -6.0)
    good_logits[:, 2, 2:4, 2:4, 2:4] = 6.0
    bad_logits = torch.full_like(target, -6.0)
    empty_target = torch.zeros_like(target)

    good_weight = mednext_causal_train._registered_modality_error_weight(good_logits, target, args)
    bad_weight = mednext_causal_train._registered_modality_error_weight(bad_logits, target, args)
    empty_weight = mednext_causal_train._registered_modality_error_weight(bad_logits, empty_target, args)

    assert bad_weight > good_weight
    assert bad_weight <= torch.tensor(2.0)
    assert torch.allclose(empty_weight, torch.tensor(1.0))


def test_registered_modality_seg_applies_error_weight() -> None:
    common = {
        "lambda_registered_modality_seg": 1.0,
        "lambda_registered_modality_consistency": 0.0,
        "lambda_registered_modality_region_consistency": 0.0,
        "lambda_registered_modality_wt_consistency": 0.0,
        "lambda_registered_modality_disease_invariance": 0.0,
        "lambda_region_loss": 1.0,
        "seg_loss_mode": "bce_dice",
        "channel_loss_weights": "1.0,1.0,1.0",
        "region_loss_weights": "1.0,1.0,1.0",
        "registered_modality_channel_loss_weights": "",
        "registered_modality_region_loss_weights": "",
        "lambda_volume_prior_loss": 0.0,
        "distill_channel_weights": "1.0,1.0,1.0",
        "registered_modality_small_lesion_emphasis": 0.0,
        "registered_modality_error_emphasis": 1.0,
        "registered_modality_error_region_weights": "0.25,1.0,2.0",
        "registered_modality_error_max_weight": 2.0,
    }
    args = type("Args", (), common)()
    registered_logits = torch.zeros(1, 3, 6, 6, 6)
    target = torch.zeros_like(registered_logits)
    target[:, 2, 2:4, 2:4, 2:4] = 1.0
    good_logits = torch.full_like(target, -6.0)
    good_logits[:, 2, 2:4, 2:4, 2:4] = 6.0
    bad_logits = torch.full_like(target, -6.0)

    good = mednext_causal_train._registered_modality_terms(
        {"logits": good_logits},
        {"logits": registered_logits},
        target,
        args,
    )
    bad = mednext_causal_train._registered_modality_terms(
        {"logits": bad_logits},
        {"logits": registered_logits},
        target,
        args,
    )

    assert bad["registered_modality_seg"] > good["registered_modality_seg"]
    assert bad["registered_modality_seg_region"] > good["registered_modality_seg_region"]


def test_registered_modality_view_advantage_distillation_teaches_weaker_view() -> None:
    args = type(
        "Args",
        (),
        {
            "registered_modality_view_advantage_region_weights": "0.25,1.0,2.0",
            "registered_modality_view_advantage_margin": 0.01,
        },
    )()
    target = torch.zeros(1, 3, 6, 6, 6)
    target[:, 2, 2:4, 2:4, 2:4] = 1.0
    native_logits = torch.full_like(target, -6.0, requires_grad=True)
    registered_logits = torch.full_like(target, -6.0, requires_grad=True)
    with torch.no_grad():
        registered_logits[:, 2, 2:4, 2:4, 2:4] = 6.0

    loss = mednext_causal_train._registered_modality_view_advantage_distillation_loss(
        native_logits,
        registered_logits,
        target,
        args,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss > torch.tensor(0.0)
    assert native_logits.grad is not None
    assert native_logits.grad.abs().sum() > 0
    assert registered_logits.grad is None or torch.allclose(
        registered_logits.grad,
        torch.zeros_like(registered_logits.grad),
    )


def test_registered_modality_view_advantage_margin_suppresses_ties() -> None:
    args = type(
        "Args",
        (),
        {
            "registered_modality_view_advantage_region_weights": "0.25,1.0,2.0",
            "registered_modality_view_advantage_margin": 1.0,
        },
    )()
    target = torch.zeros(1, 3, 4, 4, 4)
    native_logits = torch.zeros_like(target)
    registered_logits = torch.ones_like(target) * 0.1

    loss = mednext_causal_train._registered_modality_view_advantage_distillation_loss(
        native_logits,
        registered_logits,
        target,
        args,
    )

    assert torch.allclose(loss, torch.tensor(0.0))


def test_registered_modality_terms_include_view_advantage_distillation() -> None:
    args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 0.0,
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 0.0,
            "lambda_registered_modality_wt_consistency": 0.0,
            "lambda_registered_modality_view_advantage_distillation": 0.4,
            "lambda_registered_modality_disease_invariance": 0.0,
            "registered_modality_view_advantage_region_weights": "0.25,1.0,2.0",
            "registered_modality_view_advantage_margin": 0.01,
        },
    )()
    target = torch.zeros(1, 3, 6, 6, 6)
    target[:, 2, 2:4, 2:4, 2:4] = 1.0
    native_logits = torch.full_like(target, -6.0)
    registered_logits = torch.full_like(target, -6.0)
    registered_logits[:, 2, 2:4, 2:4, 2:4] = 6.0

    terms = mednext_causal_train._registered_modality_terms(
        {"logits": native_logits},
        {"logits": registered_logits},
        target,
        args,
    )

    assert "registered_modality_view_advantage_distillation" in terms
    assert terms["registered_modality_view_advantage_distillation"] > torch.tensor(0.0)
    assert torch.allclose(
        mednext_causal_train._weighted_total(terms, args),
        0.4 * terms["registered_modality_view_advantage_distillation"],
    )


def test_registered_modality_fusion_seg_supervises_inference_fusion() -> None:
    args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 0.0,
            "lambda_registered_modality_fusion_seg": 0.5,
            "registered_modality_fusion_mode": "mean-probs",
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 0.0,
            "lambda_registered_modality_wt_consistency": 0.0,
            "lambda_registered_modality_view_advantage_distillation": 0.0,
            "lambda_registered_modality_disease_invariance": 0.0,
            "lambda_region_loss": 1.0,
            "seg_loss_mode": "bce_dice",
            "channel_loss_weights": "1.0,1.0,1.0",
            "region_loss_weights": "1.0,1.0,1.0",
            "registered_modality_channel_loss_weights": "",
            "registered_modality_region_loss_weights": "",
            "lambda_volume_prior_loss": 0.0,
            "distill_channel_weights": "1.0,1.0,1.0",
            "registered_modality_small_lesion_emphasis": 0.0,
            "registered_modality_error_emphasis": 0.0,
        },
    )()
    target = torch.zeros(1, 3, 6, 6, 6)
    target[:, 2, 2:4, 2:4, 2:4] = 1.0
    native_logits = torch.full_like(target, -3.0)
    registered_logits = torch.full_like(target, -3.0)
    registered_logits[:, 2, 2:4, 2:4, 2:4] = 3.0

    terms = mednext_causal_train._registered_modality_terms(
        {"logits": native_logits},
        {"logits": registered_logits},
        target,
        args,
    )

    assert "registered_modality_fusion_seg" in terms
    assert "registered_modality_fusion_seg_region" in terms
    assert torch.isfinite(terms["registered_modality_fusion_seg"])
    assert torch.isfinite(terms["registered_modality_fusion_seg_region"])
    assert torch.allclose(
        mednext_causal_train._weighted_total(terms, args),
        0.5 * (terms["registered_modality_fusion_seg"] + terms["registered_modality_fusion_seg_region"]),
    )


def test_registered_modality_training_is_enabled_by_structural_terms() -> None:
    fusion_args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 0.0,
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 0.0,
            "lambda_registered_modality_wt_consistency": 0.0,
            "lambda_registered_modality_fusion_seg": 0.1,
            "lambda_registered_modality_view_advantage_distillation": 0.0,
            "lambda_registered_modality_disease_invariance": 0.0,
        },
    )()
    advantage_args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 0.0,
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 0.0,
            "lambda_registered_modality_wt_consistency": 0.0,
            "lambda_registered_modality_fusion_seg": 0.0,
            "lambda_registered_modality_view_advantage_distillation": 0.1,
            "lambda_registered_modality_disease_invariance": 0.0,
        },
    )()

    assert mednext_causal_train._uses_registered_modality_training(fusion_args)
    assert mednext_causal_train._uses_registered_modality_training(advantage_args)


def test_hard_case_sampler_weights_prioritize_nonempty_small_targets() -> None:
    class MaskDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.masks = []
            small = torch.zeros(3, 10, 10, 10)
            small[2, 0, 0, 0] = 1.0
            large = torch.zeros_like(small)
            large[0, :6] = 1.0
            large[2, :6] = 1.0
            empty = torch.zeros_like(small)
            self.masks = [small, large, empty]

        def __len__(self) -> int:
            return len(self.masks)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"mask": self.masks[index]}

    args = type(
        "Args",
        (),
        {
            "hard_case_sampler_emphasis": 1.0,
            "hard_case_sampler_reference_fractions": "0.2,0.1,0.05",
            "hard_case_sampler_region_weights": "0.5,1.0,2.0",
            "hard_case_sampler_max_weight": 2.0,
        },
    )()

    weights = mednext_causal_train._hard_case_sampler_weights(MaskDataset(), args)

    assert weights is not None
    assert weights[0] > weights[1]
    assert torch.allclose(weights[1], torch.tensor(1.0, dtype=torch.double))
    assert torch.allclose(weights[2], torch.tensor(1.0, dtype=torch.double))


def test_hard_case_sampler_uses_primary_masks_for_registered_pairs() -> None:
    class MaskDataset(torch.utils.data.Dataset):
        metadata_encoder = "metadata"

        def __init__(self, small_first: bool) -> None:
            small = torch.zeros(3, 10, 10, 10)
            small[2, 0, 0, 0] = 1.0
            large = torch.zeros_like(small)
            large[0, :6] = 1.0
            large[2, :6] = 1.0
            self.masks = [small, large] if small_first else [large, large]
            self.case_ids = ["A", "B"]

        def __len__(self) -> int:
            return len(self.masks)

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "case_id": self.case_ids[index],
                "image": torch.zeros(1),
                "mask": self.masks[index],
            }

    args = type(
        "Args",
        (),
        {
            "hard_case_sampler_emphasis": 1.0,
            "hard_case_sampler_reference_fractions": "0.2,0.1,0.05",
            "hard_case_sampler_region_weights": "0.5,1.0,2.0",
            "hard_case_sampler_max_weight": 2.0,
        },
    )()
    paired = mednext_causal_train.RegisteredModalityPairDataset(
        MaskDataset(small_first=True),
        MaskDataset(small_first=False),
    )

    weights = mednext_causal_train._hard_case_sampler_weights(paired, args)

    assert weights is not None
    assert weights[0] > weights[1]


def test_brats_causal_loader_uses_hard_case_sampler_when_requested() -> None:
    class MaskDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            small = torch.zeros(3, 10, 10, 10)
            small[2, 0, 0, 0] = 1.0
            large = torch.zeros_like(small)
            large[0, :6] = 1.0
            large[2, :6] = 1.0
            self.masks = [small, large]

        def __len__(self) -> int:
            return len(self.masks)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"mask": self.masks[index], "image": torch.zeros(1)}

    args = type(
        "Args",
        (),
        {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "seed": 7,
            "hard_case_sampler_emphasis": 1.0,
            "hard_case_sampler_reference_fractions": "0.2,0.1,0.05",
            "hard_case_sampler_region_weights": "0.5,1.0,2.0",
            "hard_case_sampler_max_weight": 2.0,
            "hard_case_sampler_epoch_multiplier": 2.0,
        },
    )()

    train_loader = mednext_causal_brats_train._make_causal_loader(MaskDataset(), args, shuffle=True)
    bank_loader = mednext_causal_brats_train._make_causal_loader(MaskDataset(), args, shuffle=False)

    assert isinstance(train_loader.sampler, torch.utils.data.WeightedRandomSampler)
    assert len(train_loader.sampler) == 4
    assert not isinstance(bank_loader.sampler, torch.utils.data.WeightedRandomSampler)


def test_brats_h5_hard_case_sampler_weights_use_raw_masks(tmp_path) -> None:
    def write_mask(path, mask) -> None:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("mask", data=mask)

    small_path = tmp_path / "small.h5"
    large_path = tmp_path / "large.h5"
    empty_path = tmp_path / "empty.h5"
    small = np.zeros((10, 10, 3), dtype=np.uint8)
    small[0, 0, 2] = 1
    large = np.zeros_like(small)
    large[:6, :, 0] = 1
    large[:6, :, 2] = 1
    empty = np.zeros_like(small)
    write_mask(small_path, small)
    write_mask(large_path, large)
    write_mask(empty_path, empty)

    class RawBraTSDataset(torch.utils.data.Dataset):
        data_root = None
        path_col = "path"
        slice_col = "slice"
        mask_key = "mask"

        def __init__(self) -> None:
            self.volumes = [
                (0, pd.DataFrame([{"path": str(small_path), "slice": 0}])),
                (1, pd.DataFrame([{"path": str(large_path), "slice": 0}])),
                (2, pd.DataFrame([{"path": str(empty_path), "slice": 0}])),
            ]

        def __len__(self) -> int:
            return len(self.volumes)

    args = type(
        "Args",
        (),
        {
            "hard_case_sampler_emphasis": 1.0,
            "hard_case_sampler_reference_fractions": "0.2,0.1,0.05",
            "hard_case_sampler_region_weights": "0.5,1.0,2.0",
            "hard_case_sampler_max_weight": 2.0,
        },
    )()

    weights = mednext_causal_brats_train._brats_h5_hard_case_sampler_weights(RawBraTSDataset(), args)

    assert weights is not None
    assert weights[0] > weights[1]
    assert torch.allclose(weights[1], torch.tensor(1.0, dtype=torch.double))
    assert torch.allclose(weights[2], torch.tensor(1.0, dtype=torch.double))


def test_brats_h5_hard_case_sampler_uses_metadata_fractions_without_h5() -> None:
    class MetadataBraTSDataset(torch.utils.data.Dataset):
        data_root = None
        path_col = "path"
        slice_col = "slice"
        mask_key = "mask"

        def __init__(self) -> None:
            self.volumes = [
                (
                    0,
                    pd.DataFrame(
                        [
                            {
                                "path": "missing-small.h5",
                                "slice": 0,
                                "label0_pxl_cnt": 0,
                                "label1_pxl_cnt": 0,
                                "label2_pxl_cnt": 1,
                                "background_ratio": 0.99,
                            }
                        ]
                    ),
                ),
                (
                    1,
                    pd.DataFrame(
                        [
                            {
                                "path": "missing-large.h5",
                                "slice": 0,
                                "label0_pxl_cnt": 20,
                                "label1_pxl_cnt": 20,
                                "label2_pxl_cnt": 5,
                                "background_ratio": 0.55,
                            }
                        ]
                    ),
                ),
                (
                    2,
                    pd.DataFrame(
                        [
                            {
                                "path": "missing-empty.h5",
                                "slice": 0,
                                "label0_pxl_cnt": 0,
                                "label1_pxl_cnt": 0,
                                "label2_pxl_cnt": 0,
                                "background_ratio": 1.0,
                            }
                        ]
                    ),
                ),
            ]

        def __len__(self) -> int:
            return len(self.volumes)

    args = type(
        "Args",
        (),
        {
            "hard_case_sampler_emphasis": 1.0,
            "hard_case_sampler_reference_fractions": "0.2,0.1,0.05",
            "hard_case_sampler_region_weights": "0.5,1.0,2.0",
            "hard_case_sampler_max_weight": 2.0,
        },
    )()

    weights = mednext_causal_brats_train._brats_h5_hard_case_sampler_weights(MetadataBraTSDataset(), args)

    assert weights is not None
    assert weights[0] > weights[1]
    assert torch.allclose(weights[1], torch.tensor(1.0, dtype=torch.double))
    assert torch.allclose(weights[2], torch.tensor(1.0, dtype=torch.double))


def test_brats_h5_raw_mask_slice_region_counts_support_native_formats() -> None:
    channel_last = np.zeros((4, 4, 3), dtype=np.uint8)
    channel_last[0, 0, 0] = 1
    channel_last[1, 1, 1] = 1
    channel_last[2, 2, 2] = 1

    counts, voxels = mednext_causal_brats_train._raw_mask_slice_region_counts(channel_last)

    assert voxels == 16
    assert counts.tolist() == [3.0, 2.0, 1.0]

    label_map = np.zeros((4, 4), dtype=np.uint8)
    label_map[0, 0] = 1
    label_map[1, 1] = 2
    label_map[2, 2] = 4

    counts, voxels = mednext_causal_brats_train._raw_mask_slice_region_counts(label_map)

    assert voxels == 16
    assert counts.tolist() == [3.0, 2.0, 1.0]


def test_brats_h5_hard_case_sampler_reuses_cached_raw_mask_fractions(tmp_path) -> None:
    mask_path = tmp_path / "small.h5"
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[0, 0, 2] = 1
    with h5py.File(mask_path, "w") as handle:
        handle.create_dataset("mask", data=mask)

    class RawBraTSDataset(torch.utils.data.Dataset):
        csv_path = tmp_path / "brats.csv"
        data_root = None
        path_col = "path"
        slice_col = "slice"
        mask_key = "mask"

        def __init__(self) -> None:
            self.volumes = [(7, pd.DataFrame([{"path": str(mask_path), "slice": 0}]))]

        def __len__(self) -> int:
            return len(self.volumes)

    args = type(
        "Args",
        (),
        {
            "disk_cache_dir": str(tmp_path / "cache"),
            "hard_case_sampler_emphasis": 1.0,
            "hard_case_sampler_reference_fractions": "0.2,0.1,0.05",
            "hard_case_sampler_region_weights": "0.5,1.0,2.0",
            "hard_case_sampler_max_weight": 2.0,
        },
    )()
    dataset = RawBraTSDataset()

    first = mednext_causal_brats_train._brats_h5_hard_case_sampler_weights(dataset, args)
    mask_path.unlink()
    second = mednext_causal_brats_train._brats_h5_hard_case_sampler_weights(dataset, args)

    assert first is not None
    assert second is not None
    assert torch.allclose(first, second)


def test_registered_modality_region_consistency_uses_region_weights() -> None:
    args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 0.0,
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 1.0,
            "lambda_registered_modality_disease_invariance": 0.0,
            "registered_modality_region_consistency_weights": "1.0,1.0,4.0",
        },
    )()
    uniform_args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 0.0,
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 1.0,
            "lambda_registered_modality_disease_invariance": 0.0,
            "registered_modality_region_consistency_weights": "1.0,1.0,1.0",
        },
    )()
    native_logits = torch.zeros(1, 3, 3, 3, 3)
    registered_logits = torch.zeros_like(native_logits)
    native_logits[:, 2] = 2.0
    registered_logits[:, 0] = 2.0
    outputs = {"logits": native_logits}
    registered_outputs = {"logits": registered_logits}
    target = torch.zeros_like(native_logits)

    weighted = mednext_causal_train._registered_modality_terms(outputs, registered_outputs, target, args)
    uniform = mednext_causal_train._registered_modality_terms(outputs, registered_outputs, target, uniform_args)

    assert "registered_modality_region_consistency" in weighted
    assert torch.isfinite(weighted["registered_modality_region_consistency"])
    assert weighted["registered_modality_region_consistency"] > uniform["registered_modality_region_consistency"]
    assert torch.allclose(
        mednext_causal_train._weighted_total(weighted, args),
        weighted["registered_modality_region_consistency"],
    )


def test_registered_modality_wt_consistency_penalizes_wt_drift() -> None:
    args = type(
        "Args",
        (),
        {
            "lambda_registered_modality_seg": 0.0,
            "lambda_registered_modality_consistency": 0.0,
            "lambda_registered_modality_region_consistency": 0.0,
            "lambda_registered_modality_wt_consistency": 0.3,
            "lambda_registered_modality_disease_invariance": 0.0,
        },
    )()
    native_logits = torch.full((1, 3, 4, 4, 4), -4.0)
    native_logits[:, 0, 1:3, 1:3, 1:3] = 4.0
    registered_logits = native_logits.clone()
    drifted_registered_logits = torch.full_like(native_logits, -4.0)

    aligned = mednext_causal_train._registered_modality_terms(
        {"logits": native_logits},
        {"logits": registered_logits},
        torch.zeros_like(native_logits),
        args,
    )
    drifted = mednext_causal_train._registered_modality_terms(
        {"logits": native_logits},
        {"logits": drifted_registered_logits},
        torch.zeros_like(native_logits),
        args,
    )

    assert "registered_modality_wt_consistency" in drifted
    assert torch.isfinite(drifted["registered_modality_wt_consistency"])
    assert drifted["registered_modality_wt_consistency"] > aligned["registered_modality_wt_consistency"]
    assert torch.allclose(
        mednext_causal_train._weighted_total(
            {"registered_modality_wt_consistency": torch.tensor(5.0)},
            args,
        ),
        torch.tensor(1.5),
    )


def test_brats_causal_builder_forwards_router_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_causal_mednext(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(mednext_causal_brats_train, "build_causal_mednext", fake_build_causal_mednext)
    args = type(
        "Args",
        (),
        {
            "model_id": "S",
            "kernel_size": 3,
            "latent_dim": 8,
            "base_channels": 4,
            "modulation_scale": 0.1,
            "causal_residual_scale": 0.2,
            "contrastive_dim": 8,
            "spatial_refiner_scale": 0.25,
            "region_fusion_scale": 0.0,
            "prototype_dim": 8,
            "prototype_fusion_scale": 0.0,
            "prototype_temperature": 0.1,
            "category_confounder_scale": 0.0,
            "category_confounder_temperature": 0.2,
            "modality_prior_scale": 0.0,
            "logit_calibration_scale": 0.0,
            "cascade_refiner_scale": 0.1,
            "frontdoor_mediator_scale": 0.0,
            "frontdoor_residual_scale": 0.25,
            "use_causal_mediator_router": True,
            "use_nested_causal_intervention": True,
            "nested_causal_gate_scale": 0.4,
            "region_causal_bottleneck_scale": 0.0,
            "region_causal_background_leak": 0.05,
            "region_causal_base": "factual",
            "region_causal_mask_source": "factual",
            "region_volume_scale": 1000.0,
            "et_volume_veto_scale": 0.0,
            "et_volume_veto_multiplier": 4.0,
            "et_volume_veto_min_fraction": 5e-4,
            "et_volume_veto_max_bias": 4.0,
        },
    )()

    mednext_causal_brats_train._build_model(args)

    assert captured["use_causal_mediator_router"] is True
    assert captured["use_nested_causal_intervention"] is True
    assert captured["nested_causal_gate_scale"] == 0.4


def test_brats_causal_eval_builder_forwards_router_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def load_compatible_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
            captured["loaded_state"] = state
            return {"missing_keys": [], "unexpected_keys": [], "skipped_shape_keys": []}

    def fake_build_causal_mednext(**kwargs: object) -> FakeModel:
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(mednext_causal_brats_eval, "build_causal_mednext", fake_build_causal_mednext)
    checkpoint = {
        "config": {
            "model_id": "S",
            "kernel_size": 3,
            "latent_dim": 16,
            "base_channels": 4,
            "use_causal_mediator_router": True,
            "use_nested_causal_intervention": True,
            "nested_causal_gate_scale": 0.4,
        },
        "model": {"some_weight": torch.tensor(1.0)},
    }
    args = type(
        "Args",
        (),
        {
            "model_id": None,
            "kernel_size": None,
            "latent_dim": None,
            "base_channels": None,
            "region_volume_scale": None,
            "et_volume_veto_scale": None,
            "et_volume_veto_multiplier": None,
            "et_volume_veto_min_fraction": None,
            "et_volume_veto_max_bias": None,
        },
    )()

    model = mednext_causal_brats_eval._build_model(checkpoint, args)

    assert isinstance(model, FakeModel)
    assert captured["use_causal_mediator_router"] is True
    assert captured["use_nested_causal_intervention"] is True
    assert captured["nested_causal_gate_scale"] == 0.4
    assert captured["loaded_state"] is checkpoint["model"]


def test_brats_case_record_writes_plausibility_support_metrics() -> None:
    logits = torch.full((1, 3, 4, 4, 4), -4.0)
    target = torch.ones(1, 3, 4, 4, 4)
    batch = {
        "case_id": ["volume_7"],
        "volume": torch.tensor([7]),
        "path": ["volume_7_slice_0.h5"],
    }

    record = mednext_causal_brats_eval._brats_case_record(
        batch,
        0,
        target,
        0.5,
        {"factual": logits},
        plausibility_region_thresholds=(
            {"WT": 0.5, "TC": 0.5, "ET": 0.5},
            {"WT": 0.01, "TC": 0.01, "ET": 0.01},
            0.0,
            0.9,
            0.0,
            0.1,
            0.1,
        ),
    )

    prefix = "factual_plausibility_region_calibrated"
    assert record["case_id"] == "volume_7"
    assert record["volume"] == 7
    assert record[f"{prefix}/plausibility/base_WT_pred_foreground_ratio_mean"] == 0.0
    assert record[f"{prefix}/plausibility/base_TC_pred_foreground_ratio_mean"] == 0.0
    assert record[f"{prefix}/plausibility/low_threshold_count"] == 1.0

    thresholds = fit_plausibility_support_thresholds([record], prefix=prefix)
    assert thresholds["plausibility/support_case_count"] == 1.0


def test_output_bias_initialization_matches_mask_prevalence() -> None:
    class TinyMaskDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            mask = torch.zeros(3, 2, 2, 2)
            mask[0, 0, 0, 0] = 1.0
            mask[1, 0, 0, :] = 1.0
            mask[2, 0] = 1.0
            return {"image": torch.zeros(4, 2, 2, 2), "mask": mask}

    model = build_mednext_segmenter(model_id="S", kernel_size=3, base_channels=4, deep_supervision=True)
    loader = torch.utils.data.DataLoader(TinyMaskDataset(), batch_size=1)
    args = type(
        "Args",
        (),
        {
            "init_output_bias_from_data": True,
            "output_bias_init_batches": 1,
            "output_bias_min_prob": 1e-4,
            "output_bias_max_prob": 0.9,
            "output_bias_prior_strength": 1.0,
        },
    )()

    report = initialize_output_bias_from_loader(model, loader, args)
    expected = torch.logit(torch.tensor([0.125, 0.25, 0.5]))

    assert set(report["updated_heads"]) == {"out0", "out1", "out2", "out3", "out4"}
    assert torch.allclose(model.out0.bias.detach(), expected, atol=1e-6)


def test_initialize_model_from_checkpoint_loads_baseline_weights(tmp_path) -> None:
    source = build_mednext_segmenter(model_id="S", kernel_size=3, base_channels=4, deep_supervision=False)
    with torch.no_grad():
        source.out0.bias.fill_(0.25)
    path = tmp_path / "best.pt"
    torch.save({"epoch": 3, "model": source.state_dict()}, path)

    target = build_mednext_segmenter(model_id="S", kernel_size=3, base_channels=4, deep_supervision=False)
    with torch.no_grad():
        target.out0.bias.zero_()

    report = initialize_model_from_checkpoint(target, path)

    assert report["checkpoint"] == str(path)
    assert report["epoch"] == 3
    assert report["loaded_keys"] == len(source.state_dict())
    assert torch.allclose(target.out0.bias.detach(), torch.full_like(target.out0.bias, 0.25))


def test_causal_mednext_eval_context_bank_is_skipped_for_factual_only_mode() -> None:
    adjustment_off = type("Args", (), {"adjustment_contexts": 0, "context_bank_size": 64})()
    empty_bank = type("Args", (), {"adjustment_contexts": 4, "context_bank_size": 0})()
    adjusted = type("Args", (), {"adjustment_contexts": 4, "context_bank_size": 64})()

    assert not _should_build_brats_context_bank(adjustment_off)
    assert not _should_build_brats_context_bank(empty_bank)
    assert _should_build_brats_context_bank(adjusted)
    assert not _should_build_utsw_context_bank(adjustment_off)
    assert not _should_build_utsw_context_bank(empty_bank)
    assert _should_build_utsw_context_bank(adjusted)


def test_disk_cached_dataset_reuses_preprocessed_items(tmp_path) -> None:
    class CountingDataset(torch.utils.data.Dataset):
        metadata_encoder = "delegated"

        def __init__(self) -> None:
            self.calls = 0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            self.calls += 1
            return {"image": torch.full((1,), float(index)), "case_id": "case0"}

    dataset = CountingDataset()
    cached = DiskCachedDataset(dataset, tmp_path, "toy cache", {"v": 1})

    first = cached[0]
    second = cached[0]

    assert dataset.calls == 1
    assert torch.equal(first["image"], second["image"])
    assert second["case_id"] == "case0"
    assert cached.metadata_encoder == "delegated"
    assert maybe_disk_cache_dataset(dataset, None, "off") is dataset


def test_disk_cached_dataset_rebuilds_corrupt_current_item(tmp_path) -> None:
    class CountingDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.calls = 0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            self.calls += 1
            return {"image": torch.full((1,), float(self.calls)), "case_id": "case0"}

    dataset = CountingDataset()
    cached = DiskCachedDataset(dataset, tmp_path, "toy cache", {"v": 1})
    first = cached[0]
    path = cached.cache_path(0)
    path.write_bytes(b"truncated torch cache")

    rebuilt = cached[0]
    reloaded = cached[0]

    assert dataset.calls == 2
    assert torch.equal(first["image"], torch.tensor([1.0]))
    assert torch.equal(rebuilt["image"], torch.tensor([2.0]))
    assert torch.equal(reloaded["image"], rebuilt["image"])


def test_disk_cached_dataset_reuses_overlapping_volume_id_across_subsets(tmp_path) -> None:
    class VolumeDataset(torch.utils.data.Dataset):
        def __init__(self, volume_ids: list[int]) -> None:
            self.volumes = [(volume_id, object()) for volume_id in volume_ids]
            self.calls = 0

        def __len__(self) -> int:
            return len(self.volumes)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
            self.calls += 1
            volume_id = int(self.volumes[index][0])
            return {"volume": volume_id, "image": torch.full((1,), float(volume_id))}

    signature = {"dataset": "toy", "volume_size": 32}
    small = VolumeDataset([10])
    large = VolumeDataset([10, 20])

    cached_small = DiskCachedDataset(small, tmp_path, "volumes", signature)
    cached_large = DiskCachedDataset(large, tmp_path, "volumes", signature)

    first = cached_small[0]
    reused = cached_large[0]
    new_item = cached_large[1]

    assert small.calls == 1
    assert large.calls == 1
    assert reused["volume"] == first["volume"] == 10
    assert new_item["volume"] == 20
    assert len(list((tmp_path / "volumes").glob("*/*.pt"))) == 2


def test_disk_cached_dataset_warm_cache_is_resumable(tmp_path) -> None:
    class CountingDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.calls: list[int] = []

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
            self.calls.append(int(index))
            return {"volume": int(index), "image": torch.full((1, 4, 4, 4), float(index))}

    dataset = CountingDataset()
    cached = DiskCachedDataset(dataset, tmp_path, "volumes", {"dataset": "toy", "volume_size": 4})
    _ = cached[0]
    dataset.calls.clear()

    report = cached.warm_cache(start_index=0, max_items=2)

    assert dataset.calls == [1, 2]
    assert report["before"]["missing_indices"] == [1, 2, 3]
    assert report["after"]["missing_indices"] == [3]
    assert report["warmed_count"] == 2
    assert [item["index"] for item in report["warmed"]] == [1, 2]

    wrapped_report = warm_disk_cache_dataset(cached, max_items=1)

    assert wrapped_report["enabled"] is True
    assert wrapped_report["warmed_count"] == 1
    assert wrapped_report["after"]["missing_indices"] == []
    assert warm_disk_cache_dataset(dataset)["enabled"] is False


def test_disk_cached_dataset_promotes_matching_legacy_index_item(tmp_path) -> None:
    class VolumeDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.volumes = [(10, object())]
            self.calls = 0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
            self.calls += 1
            return {"volume": 10, "image": torch.full((1, 4, 4, 4), -1.0), "mask": torch.zeros(1, 4, 4, 4)}

    legacy_dir = tmp_path / "volumes" / "legacyhash"
    legacy_dir.mkdir(parents=True)
    torch.save(
        {"volume": 10, "image": torch.full((1, 4, 4, 4), 10.0), "mask": torch.zeros(1, 4, 4, 4)},
        legacy_dir / "00000000.pt",
    )

    dataset = VolumeDataset()
    cached = DiskCachedDataset(dataset, tmp_path, "volumes", {"dataset": "toy", "volume_size": 4})

    item = cached[0]
    second = cached[0]

    assert dataset.calls == 0
    assert item["volume"] == 10
    assert torch.equal(item["image"], torch.full((1, 4, 4, 4), 10.0))
    assert torch.equal(second["image"], item["image"])
    assert len(list((tmp_path / "volumes").glob("*/*.pt"))) == 2


def test_disk_cached_dataset_promotes_legacy_item_after_index_shift(tmp_path) -> None:
    class CaseDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.cases = [type("Case", (), {"name": "case_a"})(), type("Case", (), {"name": "case_b"})()]
            self.calls = 0

        def __len__(self) -> int:
            return len(self.cases)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
            self.calls += 1
            return {"case_id": self.cases[index].name, "image": torch.full((1, 4, 4, 4), -1.0)}

    legacy_dir = tmp_path / "cases" / "legacyhash"
    legacy_dir.mkdir(parents=True)
    torch.save({"case_id": "case_b", "image": torch.full((1, 4, 4, 4), 2.0)}, legacy_dir / "00000000.pt")

    dataset = CaseDataset()
    cached = DiskCachedDataset(dataset, tmp_path, "cases", {"dataset": "toy", "volume_size": 4})

    item = cached[1]

    assert dataset.calls == 0
    assert item["case_id"] == "case_b"
    assert torch.equal(item["image"], torch.full((1, 4, 4, 4), 2.0))


def test_disk_cached_dataset_does_not_scan_unrelated_identity_legacy_items(tmp_path, monkeypatch) -> None:
    class VolumeDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.volumes = [(1, object())]
            self.calls = 0

        def __len__(self) -> int:
            return len(self.volumes)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
            self.calls += 1
            return {"volume": int(self.volumes[index][0]), "image": torch.full((1, 4, 4, 4), 10.0)}

    legacy_dir = tmp_path / "volumes" / "legacy_identity"
    legacy_dir.mkdir(parents=True)
    torch.save({"volume": 999, "image": torch.full((1, 4, 4, 4), 999.0)}, legacy_dir / "volume_999-unrelated.pt")

    dataset = VolumeDataset()
    cached = DiskCachedDataset(dataset, tmp_path, "volumes", {"dataset": "toy", "volume_size": 4})
    loaded_paths: list[str] = []
    original_load = cached._load_cache_file

    def track_load(path):
        loaded_paths.append(Path(path).name)
        return original_load(path)

    monkeypatch.setattr(cached, "_load_cache_file", track_load)

    item = cached[0]

    assert dataset.calls == 1
    assert item["volume"] == 1
    assert torch.equal(item["image"], torch.full((1, 4, 4, 4), 10.0))
    assert loaded_paths == []


def test_disk_cached_dataset_rejects_stale_legacy_index_item(tmp_path) -> None:
    class VolumeDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.volumes = [(10, object())]
            self.calls = 0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
            self.calls += 1
            return {"volume": 10, "image": torch.full((1, 4, 4, 4), 10.0), "mask": torch.zeros(1, 4, 4, 4)}

    legacy_dir = tmp_path / "volumes" / "legacy_identity"
    legacy_dir.mkdir(parents=True)
    torch.save(
        {"volume": 99, "image": torch.full((1, 4, 4, 4), 99.0), "mask": torch.zeros(1, 4, 4, 4)},
        legacy_dir / "00000000.pt",
    )
    shape_dir = tmp_path / "volumes" / "legacy_shape"
    shape_dir.mkdir(parents=True)
    torch.save(
        {"volume": 10, "image": torch.full((1, 3, 3, 3), 10.0), "mask": torch.zeros(1, 3, 3, 3)},
        shape_dir / "00000000.pt",
    )

    dataset = VolumeDataset()
    cached = DiskCachedDataset(dataset, tmp_path, "volumes", {"dataset": "toy", "volume_size": 4})

    item = cached[0]

    assert dataset.calls == 1
    assert item["volume"] == 10
    assert torch.equal(item["image"], torch.full((1, 4, 4, 4), 10.0))


def test_brats_causal_trainer_reuses_baseline_cache_signature() -> None:
    args = type(
        "Args",
        (),
        {
            "data_root": "data/brats/root",
            "volume_size": 32,
            "crop_margin": 8,
            "path_col": "path",
            "volume_col": "volume",
            "slice_col": "slice",
            "h5_image_key": "image",
            "h5_mask_key": "mask",
        },
    )()

    baseline = _brats_baseline_cache_signature("train.csv", args, "train")
    causal = _brats_causal_cache_signature("train.csv", args, "train")

    assert causal == baseline
    assert causal["dataset"] == "brats_h5"


def test_region_threshold_metrics_match_global_threshold_when_equal() -> None:
    logits = torch.randn(2, 3, 4, 4, 4)
    target = torch.randint(0, 2, (2, 3, 4, 4, 4)).float()

    raw = brats_region_metrics(logits, target, threshold=0.5)
    calibrated = brats_region_metrics_from_thresholds(logits, target, {"WT": 0.5, "TC": 0.5, "ET": 0.5})

    for key in ("brats/mean_dice", "brats/WT/dice", "brats/TC/dice", "brats/ET/dice"):
        assert abs(raw[key] - calibrated[key]) < 1e-6


def test_region_threshold_sweep_can_select_region_specific_thresholds() -> None:
    target = torch.zeros(1, 3, 1, 1, 4)
    target[:, 1, :, :, :3] = 1.0
    target[:, 2, :, :, 3:] = 1.0
    probs = torch.full_like(target, 0.1)
    probs[:, 1, :, :, :3] = 0.35
    probs[:, 2, :, :, :3] = 0.45
    probs[:, 2, :, :, 3:] = 0.65
    logits = torch.logit(probs.clamp(1e-4, 1.0 - 1e-4))

    sweep = BratsRegionThresholdSweep([0.3, 0.5, 0.7])
    sweep.update(logits, target)
    summary = sweep.summary()

    assert summary["threshold/WT"] == 0.3
    assert summary["threshold/ET"] == 0.5
    assert summary["brats/mean_dice"] > brats_region_metrics(logits, target, threshold=0.5)["brats/mean_dice"]
    assert parse_region_thresholds("WT=0.3,TC=0.5,ET=0.7") == {"WT": 0.3, "TC": 0.5, "ET": 0.7}


def test_adaptive_region_threshold_sweep_selects_low_confidence_rule() -> None:
    target = torch.zeros(1, 3, 1, 1, 4)
    target[:, 1, :, :, :2] = 1.0
    probs = torch.full_like(target, 0.01)
    probs[:, 1, :, :, :2] = 0.1
    logits = torch.logit(probs.clamp(1e-4, 1.0 - 1e-4))

    sweep = BratsAdaptiveRegionThresholdSweep(
        {"WT": 0.4, "TC": 0.4, "ET": 0.8},
        low_candidates=[0.05, 0.2],
        wt_ratio_candidates=[0.0, 0.01],
    )
    sweep.update(logits, target)
    summary = sweep.summary()

    assert summary["adaptive/wt_ratio_threshold"] == 0.01
    assert summary["adaptive/low_threshold/WT"] == 0.05
    assert summary["adaptive/low_threshold_fraction"] == 1.0
    assert summary["brats/mean_dice"] > 0.99
    assert parse_fraction_candidates("0,0.01,0.01") == [0.0, 0.01]


def test_plausibility_thresholds_switch_only_out_of_support_patterns() -> None:
    target = torch.zeros(3, 3, 1, 1, 100)
    target[0, 1, :, :, :10] = 1.0
    target[1, 1, :, :, :1] = 1.0
    target[2, 1, :, :, :10] = 1.0
    probs = torch.full_like(target, 0.01)
    probs[0, 1, :, :, :10] = 0.1
    probs[1, 1, :, :, :1] = 0.6
    probs[2, 1, :, :, :10] = 0.6
    logits = torch.logit(probs.clamp(1e-4, 1.0 - 1e-4))

    metrics = brats_region_metrics_from_plausibility_thresholds(
        logits,
        target,
        {"WT": 0.4, "TC": 0.4, "ET": 0.8},
        {"WT": 0.05, "TC": 0.05, "ET": 0.5},
        low_stability_wt_ratio_threshold=0.01,
        low_stability_threshold=0.9,
        tc_collapse_wt_ratio_min=0.005,
        tc_collapse_wt_ratio_max=0.02,
        tc_collapse_tc_ratio_threshold=0.001,
        stability_scores=torch.tensor([0.5, 0.99, 0.99]),
    )

    assert metrics["plausibility/low_threshold_count"] == 2.0
    assert abs(metrics["plausibility/low_stability_gate_fraction"] - 1.0 / 3.0) < 1e-6
    assert abs(metrics["plausibility/tc_collapse_gate_fraction"] - 1.0 / 3.0) < 1e-6
    assert metrics["brats/mean_dice"] > 0.6


def test_fit_plausibility_support_thresholds_uses_train_val_boundaries() -> None:
    prefix = "registered_tta_plausibility_region_calibrated"

    def record(wt: float, tc: float, stability: float) -> dict[str, float]:
        return {
            f"{prefix}/plausibility/base_WT_pred_foreground_ratio_mean": wt,
            f"{prefix}/plausibility/base_TC_pred_foreground_ratio_mean": tc,
            f"{prefix}/plausibility/stability_score_mean": stability,
        }

    train_records = [
        record(0.006, 0.0002, 0.89),
        record(0.002, 0.0006, 0.97),
        record(0.020, 0.0100, 0.98),
    ]
    val_records = [
        record(0.011, 0.0050, 0.95),
        record(0.015, 0.0060, 0.98),
    ]

    thresholds = fit_plausibility_support_thresholds(
        [*train_records, *val_records],
        validation_records=val_records,
        low_stability_threshold=0.9,
        low_stability_wt_margin=0.95,
        tc_collapse_tc_margin=0.5,
    )

    assert abs(thresholds["plausibility/low_stability_wt_ratio_threshold"] - 0.0057) < 1e-9
    assert thresholds["plausibility/tc_collapse_wt_ratio_min"] == 0.0
    assert thresholds["plausibility/tc_collapse_wt_ratio_max"] == 0.011
    assert thresholds["plausibility/tc_collapse_tc_ratio_threshold"] == 0.0001
    assert thresholds["plausibility/support_case_count"] == 5.0
    assert thresholds["plausibility/validation_case_count"] == 2.0


def test_checkpoint_monitor_prefers_calibrated_metrics_only_when_enabled() -> None:
    metrics = {
        "brats/mean_dice": 0.1,
        "adjusted/brats/mean_dice": 0.2,
        "sweep_region_calibrated/brats/mean_dice": 0.3,
        "adjusted_sweep_region_calibrated/brats/mean_dice": 0.4,
        "registered_tta/brats/mean_dice": 0.5,
        "registered_tta_sweep_region_calibrated/brats/mean_dice": 0.6,
        "registered_consistency/brats/mean_agreement_dice": 0.7,
        "registered_consistency/region_prob_similarity": 0.8,
        "registered_consistency/stability_score": 0.9,
        "registered_consistency/region_prob_l1": 0.01,
        "selection/negative_loss": -0.05,
    }
    raw_args = type("Args", (), {"checkpoint_calibration_thresholds": None})()
    calibrated_args = type("Args", (), {"checkpoint_calibration_thresholds": "0.3,0.5,0.7"})()
    registered_args = type(
        "Args",
        (),
        {
            "checkpoint_calibration_thresholds": "0.3,0.5,0.7",
            "checkpoint_registered_modality_tta": True,
        },
    )()

    value, key = checkpoint_monitor(metrics, raw_args)
    assert value == 0.1
    assert key == "brats/mean_dice"

    value, key = checkpoint_monitor(metrics, raw_args, prefer_adjusted=True)
    assert value == 0.2
    assert key == "adjusted/brats/mean_dice"

    value, key = checkpoint_monitor(metrics, calibrated_args)
    assert value == 0.3
    assert key == "sweep_region_calibrated/brats/mean_dice"

    value, key = checkpoint_monitor(metrics, calibrated_args, prefer_adjusted=True)
    assert value == 0.4
    assert key == "adjusted_sweep_region_calibrated/brats/mean_dice"

    value, key = checkpoint_monitor(metrics, registered_args, prefer_adjusted=True)
    assert value == 0.6
    assert key == "registered_tta_sweep_region_calibrated/brats/mean_dice"

    stability_args = type(
        "Args",
        (),
        {
            "checkpoint_calibration_thresholds": "0.3,0.5,0.7",
            "checkpoint_registered_modality_tta": True,
            "checkpoint_registered_modality_selector": "stability",
        },
    )()
    value, key = checkpoint_monitor(metrics, stability_args, prefer_adjusted=True)
    assert value == 0.9
    assert key == "registered_consistency/stability_score"

    response_args = type(
        "Args",
        (),
        {
            "checkpoint_calibration_thresholds": "0.3,0.5,0.7",
            "checkpoint_registered_modality_tta": True,
            "checkpoint_registered_modality_selector": "region-prob-response",
        },
    )()
    value, key = checkpoint_monitor(metrics, response_args, prefer_adjusted=True)
    assert value == metrics["registered_consistency/region_prob_l1"]
    assert key == "registered_consistency/region_prob_l1"

    loss_args = type(
        "Args",
        (),
        {
            "checkpoint_calibration_thresholds": "0.3,0.5,0.7",
            "checkpoint_registered_modality_tta": True,
            "checkpoint_registered_modality_selector": "val-loss",
        },
    )()
    value, key = checkpoint_monitor(metrics, loss_args, prefer_adjusted=True)
    assert value == -0.05
    assert key == "selection/negative_loss"


def test_registered_modality_consistency_metrics_reward_alignment() -> None:
    native_logits = torch.full((1, 3, 3, 3, 3), -2.0)
    native_logits[:, :, 1:, 1:, 1:] = 2.0
    aligned = native_logits.clone()
    shifted = torch.full_like(native_logits, -2.0)
    shifted[:, :, :2, :2, :2] = 2.0

    aligned_metrics = registered_modality_consistency_metrics(native_logits, aligned)
    shifted_metrics = registered_modality_consistency_metrics(native_logits, shifted)

    assert aligned_metrics["registered_consistency/prob_l1"] == 0.0
    assert aligned_metrics["registered_consistency/brats/mean_agreement_dice"] == 1.0
    assert aligned_metrics["registered_consistency/stability_score"] > shifted_metrics["registered_consistency/stability_score"]


def test_causal_utsw_eval_epoch_emits_registered_tta_checkpoint_metrics(monkeypatch) -> None:
    class FakeCausalModel:
        def eval(self) -> None:
            return None

        def __call__(self, image: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "logits": image[:, :3],
                "z_d": torch.zeros(image.shape[0], 2, device=image.device),
                "z_c": torch.zeros(image.shape[0], 2, device=image.device),
            }

        def add_cite_outputs(self, outputs: dict[str, torch.Tensor], bank: object, max_negatives: int = 0) -> None:
            return None

    monkeypatch.setattr(
        mednext_causal_train,
        "_causal_loss_terms",
        lambda outputs, batch, args, proxy_layout: {"seg": outputs["logits"].sum() * 0.0},
    )
    monkeypatch.setattr(mednext_causal_train, "_weighted_total", lambda terms, args: next(iter(terms.values())))
    monkeypatch.setattr(mednext_causal_train, "_add_context_swap_outputs", lambda *args, **kwargs: None)

    image = torch.full((3, 4, 4, 4), -2.0)
    registered_image = image.clone()
    registered_image[:, 1:3, 1:3, 1:3] = 3.0
    mask = torch.zeros(3, 4, 4, 4)
    mask[:, 1:3, 1:3, 1:3] = 1.0
    loader = torch.utils.data.DataLoader(
        [{"image": image, "registered_image": registered_image, "mask": mask}],
        batch_size=1,
    )
    args = type(
        "Args",
        (),
        {
            "checkpoint_calibration_thresholds": "0.3,0.5,0.7",
            "checkpoint_registered_modality_tta": True,
            "checkpoint_registered_modality_fusion": "mean-probs",
            "adjustment_contexts": 0,
            "adversary_strength": 0.0,
            "cite_bank_negatives": 0,
            "threshold": 0.5,
            "max_val_batches": None,
        },
    )()

    summary = mednext_causal_train._run_eval_epoch(
        FakeCausalModel(),
        loader,
        torch.device("cpu"),
        args,
        context_bank=None,
        contrastive_bank=None,
        proxy_layout=None,
    )

    assert "registered_tta/brats/mean_dice" in summary
    assert "registered_tta_sweep_region_calibrated/brats/mean_dice" in summary
    assert "registered_consistency/stability_score" in summary
    assert "registered_consistency/brats/mean_agreement_dice" in summary
    assert "selection/negative_loss" in summary
    assert summary["registered_tta_sweep_region_calibrated/brats/mean_dice"] > summary["sweep_region_calibrated/brats/mean_dice"]


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


def test_et_volume_veto_is_default_off_and_suppresses_excess_et() -> None:
    logits = torch.full((1, 3, 2, 2, 2), -4.0)
    logits[:, 2] = 4.0
    low_volume_proxy = torch.zeros(1, 3)

    off_model = build_causal_mednext(
        model_id="S",
        kernel_size=3,
        base_channels=4,
        latent_dim=8,
        et_volume_veto_scale=0.0,
    )
    off_logits, off_info = off_model._apply_et_volume_veto(logits, low_volume_proxy)

    assert torch.allclose(off_logits, logits)
    assert off_info == {}

    on_model = build_causal_mednext(
        model_id="S",
        kernel_size=3,
        base_channels=4,
        latent_dim=8,
        region_volume_scale=1000.0,
        et_volume_veto_scale=2.0,
        et_volume_veto_multiplier=1.0,
        et_volume_veto_min_fraction=0.01,
        et_volume_veto_max_bias=3.0,
    )
    adjusted_logits, info = on_model._apply_et_volume_veto(logits, low_volume_proxy)

    assert torch.allclose(adjusted_logits[:, :2], logits[:, :2])
    assert torch.all(adjusted_logits[:, 2] < logits[:, 2])
    assert info["et_volume_veto_bias"].shape == (1,)
    assert torch.allclose(info["et_volume_veto_bias"], torch.tensor([3.0]))
    assert torch.allclose(info["et_volume_veto_allowed_fraction"], torch.tensor([0.01]))
    assert torch.allclose(info["et_volume_proxy_fraction"], torch.tensor([0.0]))


def test_et_volume_veto_epoch_schedule_warms_up_and_ramps() -> None:
    args = type(
        "Args",
        (),
        {
            "et_volume_veto_scale": 2.0,
            "et_volume_veto_warmup_epochs": 1,
            "et_volume_veto_ramp_epochs": 2,
        },
    )()

    assert _et_volume_veto_scale_for_epoch(args, 1) == 0.0
    assert _et_volume_veto_scale_for_epoch(args, 2) == 1.0
    assert _et_volume_veto_scale_for_epoch(args, 3) == 2.0
    assert _et_volume_veto_scale_for_epoch(args, 4) == 2.0

    no_ramp = type(
        "Args",
        (),
        {
            "et_volume_veto_scale": 2.0,
            "et_volume_veto_warmup_epochs": 1,
            "et_volume_veto_ramp_epochs": 0,
        },
    )()
    assert _et_volume_veto_scale_for_epoch(no_ramp, 1) == 0.0
    assert _et_volume_veto_scale_for_epoch(no_ramp, 2) == 2.0


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


def test_causal_mednext_init_checkpoint_loads_compatible_weights(tmp_path) -> None:
    model = build_causal_mednext(model_id="S", kernel_size=3, base_channels=4, latent_dim=8)
    source = build_causal_mednext(model_id="S", kernel_size=3, base_channels=4, latent_dim=8)
    with torch.no_grad():
        source.causal_residual_gate.bias.fill_(0.25)
    state = {key: value.clone() for key, value in source.state_dict().items()}
    state["spatial_refiner_head.0.weight"] = torch.randn(1, 1, 1, 1, 1)
    checkpoint_path = tmp_path / "causal.pt"
    torch.save({"model": state, "epoch": 3}, checkpoint_path)

    report = _load_causal_init_checkpoint(model, checkpoint_path)

    assert report["epoch"] == 3
    assert report["checkpoint"] == str(checkpoint_path)
    assert "spatial_refiner_head.0.weight" in report["skipped_shape_keys"]
    assert torch.allclose(model.causal_residual_gate.bias, torch.full_like(model.causal_residual_gate.bias, 0.25))


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


def test_registered_modality_tta_fusion_modes() -> None:
    native_prob = torch.tensor([[[[[0.2, 0.8]]]]])
    registered_prob = torch.tensor([[[[[0.6, 0.4]]]]])
    native_logits = torch.logit(native_prob)
    registered_logits = torch.logit(registered_prob)

    mean_probs = torch.sigmoid(_fuse_registered_modality_logits(native_logits, registered_logits, "mean-probs"))
    max_probs = torch.sigmoid(_fuse_registered_modality_logits(native_logits, registered_logits, "max-probs"))
    registered_only = _fuse_registered_modality_logits(native_logits, registered_logits, "registered-only")

    assert torch.allclose(mean_probs, torch.tensor([[[[[0.4, 0.6]]]]]), atol=1e-4)
    assert torch.allclose(max_probs, torch.tensor([[[[[0.6, 0.8]]]]]), atol=1e-4)
    assert torch.allclose(registered_only, registered_logits)


def test_registered_modality_stability_gated_fusion_switches_on_disagreement() -> None:
    native_prob = torch.full((2, 3, 2, 2, 2), 0.2)
    registered_prob = native_prob.clone()
    native_prob[1] = 0.1
    registered_prob[1] = 0.9
    native_logits = torch.logit(native_prob)
    registered_logits = torch.logit(registered_prob)
    stability, gate_mask = _registered_modality_stability_gate_mask(
        native_logits,
        registered_logits,
        stability_gate_threshold=0.9,
    )

    fused = torch.sigmoid(
        _fuse_registered_modality_logits(native_logits, registered_logits, "stability-gated-registered")
    )
    disabled_gate = torch.sigmoid(
        _fuse_registered_modality_logits(
            native_logits,
            registered_logits,
            "stability-gated-registered",
            stability_gate_threshold=0.0,
        )
    )

    assert stability[0] > 0.9
    assert stability[1] < 0.9
    assert gate_mask.tolist() == [False, True]
    assert torch.allclose(fused[0], native_prob[0], atol=1e-4)
    assert torch.allclose(fused[1], registered_prob[1], atol=1e-4)
    assert torch.allclose(disabled_gate[1], torch.full_like(registered_prob[1], 0.5), atol=1e-4)


def test_case_record_includes_region_calibrated_metrics() -> None:
    logits = torch.logit(torch.tensor([[[[[0.2, 0.8]]], [[[0.1, 0.1]]], [[[0.1, 0.1]]]]]))
    target = torch.tensor([[[[[0.0, 1.0]]], [[[0.0, 0.0]]], [[[0.0, 0.0]]]]])

    record = _case_record(
        "case",
        {},
        0,
        target,
        0.5,
        {"registered_tta": logits},
        region_thresholds={"WT": 0.1, "TC": 0.5, "ET": 0.5},
        adaptive_region_thresholds=(
            {"WT": 0.5, "TC": 0.5, "ET": 0.5},
            {"WT": 0.1, "TC": 0.5, "ET": 0.5},
            0.75,
        ),
    )

    assert record["registered_tta/brats/WT/dice"] > 0.99
    assert record["registered_tta_region_calibrated/brats/WT/dice"] < record["registered_tta/brats/WT/dice"]
    assert "registered_tta_region_calibrated/brats/mean_dice" in record
    assert record["registered_tta_region_calibrated/brats/WT/pred_foreground_ratio"] > 0.5
    assert record["registered_tta_adaptive_region_calibrated/adaptive/low_threshold_fraction"] == 1.0


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
