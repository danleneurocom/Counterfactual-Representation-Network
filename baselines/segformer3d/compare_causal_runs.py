from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def _row(name: str, metrics: dict[str, Any]) -> list[str]:
    return [
        name,
        _fmt(_metric(metrics, "volume/brats/mean_dice", "brats/mean_dice")),
        _fmt(_metric(metrics, "adjusted/volume/brats/mean_dice", "adjusted/brats/mean_dice")),
        _fmt(_metric(metrics, "volume/brats/WT/dice", "brats/WT/dice")),
        _fmt(_metric(metrics, "volume/brats/TC/dice", "brats/TC/dice")),
        _fmt(_metric(metrics, "volume/brats/ET/dice", "brats/ET/dice")),
        _fmt(_metric(metrics, "volume/brats/mean_hd95")),
        _fmt(_metric(metrics, "adjusted/volume/brats/mean_hd95")),
        _fmt(_metric(metrics, "intervention/context_adjustment_mean_abs_prob_shift")),
        _fmt(_metric(metrics, "overlap/nearest_context_l2_mean")),
    ]


def _recommend(reference: dict[str, Any], candidate: dict[str, Any], min_dice_gain: float, max_et_drop: float) -> str:
    ref_dice = _metric(reference, "volume/brats/mean_dice", "brats/mean_dice")
    cand_dice = _metric(candidate, "volume/brats/mean_dice", "brats/mean_dice")
    ref_et = _metric(reference, "volume/brats/ET/dice", "brats/ET/dice")
    cand_et = _metric(candidate, "volume/brats/ET/dice", "brats/ET/dice")
    ref_shift = _metric(reference, "intervention/context_adjustment_mean_abs_prob_shift") or 0.0
    cand_shift = _metric(candidate, "intervention/context_adjustment_mean_abs_prob_shift") or 0.0
    if ref_dice is None or cand_dice is None:
        return "inspect: missing factual mean Dice"
    dice_gain = cand_dice - ref_dice
    et_drop = 0.0 if ref_et is None or cand_et is None else ref_et - cand_et
    if dice_gain >= min_dice_gain and et_drop <= max_et_drop and cand_shift > ref_shift:
        return "keep: candidate improves Dice and causal response without excessive ET drop"
    if cand_shift > ref_shift and dice_gain > -min_dice_gain and et_drop <= max_et_drop:
        return "ablation-only: causal response improved, but Dice gain is not strong enough"
    return "remove-or-reduce: candidate does not justify the added mechanism"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SegFormer3D causal metric JSON files.")
    parser.add_argument("--baseline", default="runs/segformer3d_utsw_base/test_metrics.json")
    parser.add_argument("--reference", default="runs/segformer3d_utsw_causal/test_causal_metrics.json")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--min-dice-gain", type=float, default=0.002)
    parser.add_argument("--max-et-drop", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _load(args.baseline)
    reference = _load(args.reference)
    candidate = _load(args.candidate)
    header = [
        "run",
        "factual_dice",
        "adjusted_dice",
        "WT",
        "TC",
        "ET",
        "hd95",
        "adjusted_hd95",
        "context_shift",
        "overlap_mean",
    ]
    rows = [
        _row("baseline", baseline),
        _row("reference", reference),
        _row(args.candidate_name, candidate),
    ]
    print("\t".join(header))
    for row in rows:
        print("\t".join(row))
    print(_recommend(reference, candidate, args.min_dice_gain, args.max_et_drop))


if __name__ == "__main__":
    main()
