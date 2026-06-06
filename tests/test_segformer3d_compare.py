from baselines.segformer3d.compare_causal_runs import _recommend, _row


def test_compare_prefers_volume_metrics_for_final_decisions() -> None:
    metrics = {
        "brats/mean_dice": 0.99,
        "volume/brats/mean_dice": 0.75,
        "adjusted/brats/mean_dice": 0.98,
        "adjusted/volume/brats/mean_dice": 0.74,
        "brats/ET/dice": 0.97,
        "volume/brats/ET/dice": 0.71,
    }

    row = _row("candidate", metrics)

    assert row[1] == "0.750000"
    assert row[2] == "0.740000"
    assert row[5] == "0.710000"


def test_compare_removes_candidate_that_loses_volume_dice() -> None:
    reference = {
        "volume/brats/mean_dice": 0.768,
        "volume/brats/ET/dice": 0.735,
        "intervention/context_adjustment_mean_abs_prob_shift": 0.000002,
    }
    candidate = {
        "volume/brats/mean_dice": 0.740,
        "volume/brats/ET/dice": 0.711,
        "intervention/context_adjustment_mean_abs_prob_shift": 0.000013,
    }

    decision = _recommend(reference, candidate, min_dice_gain=0.002, max_et_drop=0.005)

    assert decision.startswith("remove-or-reduce")
