from crn.train import _effective_loss_config, _should_update_best


def test_effective_loss_config_warms_selected_terms_only() -> None:
    loss_config = {
        "lambda_seg": 1.0,
        "lambda_seg_adjustment": 0.4,
        "lambda_seg_cf_stability": 0.2,
        "lambda_seg_disease_swap": 0.1,
        "lambda_dis": 0.01,
    }
    effective, factor = _effective_loss_config(
        loss_config,
        epoch=2,
        warmup_config={
            "epochs": 3,
            "start_factor": 0.25,
            "keys": ["lambda_seg_adjustment", "lambda_seg_cf_stability", "lambda_seg_disease_swap", "lambda_dis"],
        },
    )

    assert factor == 0.625
    assert effective["lambda_seg"] == 1.0
    assert effective["lambda_seg_adjustment"] == 0.25
    assert effective["lambda_seg_cf_stability"] == 0.125
    assert effective["lambda_seg_disease_swap"] == 0.0625
    assert effective["lambda_dis"] == 0.00625


def test_effective_loss_config_default_warmup_includes_region_terms() -> None:
    effective, factor = _effective_loss_config(
        {
            "lambda_region_adjustment": 0.4,
            "lambda_region_cf_contrastive": 0.3,
            "lambda_region_cf_stability": 0.2,
            "lambda_region_disease_swap": 0.1,
        },
        epoch=1,
        warmup_config={"epochs": 2, "start_factor": 0.5},
    )

    assert factor == 0.5
    assert effective["lambda_region_adjustment"] == 0.2
    assert effective["lambda_region_cf_contrastive"] == 0.15
    assert effective["lambda_region_cf_stability"] == 0.1
    assert effective["lambda_region_disease_swap"] == 0.05


def test_should_update_best_uses_tiebreak_metric() -> None:
    train_config = {
        "checkpoint_metric": "sweep_best_volume/brats/mean_dice",
        "checkpoint_mode": "max",
        "checkpoint_tiebreak_metric": "sweep_best_volume/brats/mean_hd95",
        "checkpoint_tiebreak_mode": "min",
    }
    metrics = {
        "sweep_best_volume/brats/mean_dice": 0.75,
        "sweep_best_volume/brats/mean_hd95": 16.2,
    }

    should_update, primary, secondary = _should_update_best(
        metrics,
        best_primary=0.75,
        best_secondary=17.0,
        train_config=train_config,
    )

    assert should_update
    assert primary == 0.75
    assert secondary == 16.2


def test_should_not_update_when_primary_and_tiebreak_are_worse() -> None:
    train_config = {
        "checkpoint_metric": "sweep_best_volume/brats/mean_dice",
        "checkpoint_mode": "max",
        "checkpoint_tiebreak_metric": "sweep_best_volume/brats/mean_hd95",
        "checkpoint_tiebreak_mode": "min",
    }
    metrics = {
        "sweep_best_volume/brats/mean_dice": 0.74,
        "sweep_best_volume/brats/mean_hd95": 15.0,
    }

    should_update, primary, secondary = _should_update_best(
        metrics,
        best_primary=0.75,
        best_secondary=14.5,
        train_config=train_config,
    )

    assert not should_update
    assert primary == 0.74
    assert secondary == 15.0
