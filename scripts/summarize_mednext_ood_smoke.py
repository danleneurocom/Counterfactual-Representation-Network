#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REGIONS = ("WT", "TC", "ET")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _nested(data: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _metric(data: dict[str, Any] | None, key: str, section: str | None = None) -> float | None:
    value = _nested(data, section, key) if section else _nested(data, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row(
    stage: str,
    source: str,
    data: dict[str, Any] | None,
    *,
    prefix: str = "",
    section: str | None = None,
    note: str = "",
    target: float,
) -> dict[str, Any] | None:
    if data is None:
        return None
    base = f"{prefix}/brats" if prefix else "brats"
    mean = _metric(data, f"{base}/mean_dice", section=section)
    if mean is None:
        return None
    values = {
        region: _metric(data, f"{base}/{region}/dice", section=section)
        for region in REGIONS
    }
    tc = values.get("TC")
    et = values.get("ET")
    status = "pass" if mean >= target else "watch"
    if tc is not None and et is not None and tc >= target and et >= target:
        status = "pass TC/ET"
    elif tc is not None and tc >= target and et is not None and et < target:
        status = "ET watch"
    elif et is not None and et >= target and tc is not None and tc < target:
        status = "TC watch"
    return {
        "stage": stage,
        "source": source,
        "mean": mean,
        "WT": values.get("WT"),
        "TC": tc,
        "ET": et,
        "status": status,
        "note": note,
    }


def _format(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Stage | Source | Mean | WT | TC | ET | Status | Note |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {source} | {mean} | {WT} | {TC} | {ET} | {status} | {note} |".format(
                stage=row["stage"],
                source=row["source"],
                mean=_format(row["mean"]),
                WT=_format(row["WT"]),
                TC=_format(row["TC"]),
                ET=_format(row["ET"]),
                status=row["status"],
                note=row["note"],
            )
        )
    return "\n".join(lines)


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(rows, key=lambda item: float(item["mean"])) if rows else None


def build_report(args: argparse.Namespace) -> str:
    zero_shot = _load_json(args.zero_shot_json)
    adapted_eval = _load_json(args.adapted_eval_json)
    causal_epoch = _load_json(args.causal_epoch_json)
    cct_eval = _load_json(args.cct_eval_json)
    et_focused_epoch = _load_json(args.et_focused_json)
    causal_et_epoch = _load_json(args.causal_et_json)
    causal_et_fine = _load_json(args.causal_et_fine_json)
    causal_et_precision = _load_json(args.causal_et_precision_json)
    causal_et_structural = _load_json(args.causal_et_structural_json)

    rows: list[dict[str, Any]] = []
    for row in (
        _row(
            "zero-shot",
            str(args.zero_shot_json),
            zero_shot,
            note="UTSW checkpoint directly on BraTS",
            target=args.target_mean_dice,
        ),
        _row(
            "adapted raw",
            str(args.adapted_eval_json),
            adapted_eval,
            note="few-shot target adaptation",
            target=args.target_mean_dice,
        ),
        _row(
            "adapted calibrated",
            str(args.adapted_eval_json),
            adapted_eval,
            prefix="sweep_region_calibrated",
            note="region threshold sweep",
            target=args.target_mean_dice,
        ),
        _row(
            "SCM raw",
            str(args.causal_epoch_json),
            causal_epoch,
            section="val",
            note="causal branch factual validation",
            target=args.target_mean_dice,
        ),
        _row(
            "SCM adjusted",
            str(args.causal_epoch_json),
            causal_epoch,
            section="val",
            prefix="adjusted",
            note="context-adjusted validation",
            target=args.target_mean_dice,
        ),
        _row(
            "SCM adjusted calibrated",
            str(args.causal_epoch_json),
            causal_epoch,
            section="val",
            prefix="adjusted_sweep_region_calibrated",
            note="context-adjusted threshold sweep",
            target=args.target_mean_dice,
        ),
        _row(
            "ET-focused adapted",
            str(args.et_focused_json),
            et_focused_epoch,
            section="val",
            prefix="sweep_region_calibrated",
            note="ET-weighted continuation",
            target=args.target_mean_dice,
        ),
        _row(
            "ET-focused SCM adjusted",
            str(args.causal_et_json),
            causal_et_epoch,
            section="val",
            prefix="adjusted_sweep_region_calibrated",
            note="support-aware SCM continuation",
            target=args.target_mean_dice,
        ),
        _row(
            "ET-focused SCM fine calibrated",
            str(args.causal_et_fine_json),
            causal_et_fine,
            prefix="adjusted_sweep_region_calibrated",
            note="fine WT/TC/ET threshold sweep",
            target=args.target_mean_dice,
        ),
        _row(
            "ET-precision SCM TC/ET calibrated",
            str(args.causal_et_precision_json),
            causal_et_precision,
            section="val",
            prefix="adjusted_sweep_region_calibrated",
            note="TC/ET-min checkpoint objective",
            target=args.target_mean_dice,
        ),
        _row(
            "ET-precision SCM structural calibrated",
            str(args.causal_et_structural_json),
            causal_et_structural,
            prefix="adjusted_structural_region_calibrated",
            note="calibrated thresholds plus component prior",
            target=args.target_mean_dice,
        ),
        _row(
            "CCT consensus",
            str(args.cct_eval_json),
            cct_eval,
            prefix="cct_consensus",
            note="counterfactual context consensus",
            target=args.target_mean_dice,
        ),
        _row(
            "CCT gated",
            str(args.cct_eval_json),
            cct_eval,
            prefix="cct_stability_gated",
            note="instability-gated CCT",
            target=args.target_mean_dice,
        ),
        _row(
            "CCT consensus calibrated",
            str(args.cct_eval_json),
            cct_eval,
            prefix="cct_consensus_region_calibrated",
            note="fixed WT/TC/ET thresholds",
            target=args.target_mean_dice,
        ),
        _row(
            "CCT gated calibrated",
            str(args.cct_eval_json),
            cct_eval,
            prefix="cct_stability_gated_region_calibrated",
            note="fixed WT/TC/ET thresholds",
            target=args.target_mean_dice,
        ),
    ):
        if row is not None:
            rows.append(row)

    lines = ["# UTSW -> BraTS OOD Smoke Summary", ""]
    if not rows:
        lines.append("No readable OOD smoke metrics were found.")
        return "\n".join(lines)

    lines.append(_markdown_table(rows))
    lines.append("")
    best = _best_row(rows)
    if best is not None:
        lines.append(
            "Best observed row: **{stage}** with mean Dice {mean}, TC {tc}, ET {et}.".format(
                stage=best["stage"],
                mean=_format(best["mean"]),
                tc=_format(best["TC"]),
                et=_format(best["ET"]),
            )
        )
    zero = next((row for row in rows if row["stage"] == "zero-shot"), None)
    if zero is not None and best is not None and best is not zero:
        lines.append(
            "OOD recovery over zero-shot: mean {mean:+.3f}, TC {tc:+.3f}, ET {et:+.3f}.".format(
                mean=float(best["mean"]) - float(zero["mean"]),
                tc=float(best["TC"]) - float(zero["TC"]) if best["TC"] is not None and zero["TC"] is not None else 0.0,
                et=float(best["ET"]) - float(zero["ET"]) if best["ET"] is not None and zero["ET"] is not None else 0.0,
            )
        )
    lines.append("")
    lines.append(
        "Interpretation: zero-shot cross-dataset Dice is the true OOD initialization; "
        "few-shot adaptation and SCM/CCT rows measure recoverability under the domain intervention."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MedNeXt UTSW->BraTS OOD smoke metrics without importing torch.")
    parser.add_argument("--zero-shot-json", type=Path, default=Path("runs/_ood_utsw_best_on_brats_val_smoke2.json"))
    parser.add_argument(
        "--adapted-eval-json",
        type=Path,
        default=Path("runs/_ood_adapt_utsw_to_brats_v4_bias_region_e4/brats_val_smoke4_metrics.json"),
    )
    parser.add_argument(
        "--causal-epoch-json",
        type=Path,
        default=Path("runs/_ood_causal_adapt_brats_v4_e1/epoch_001.json"),
    )
    parser.add_argument(
        "--cct-eval-json",
        type=Path,
        default=Path("runs/_ood_causal_adapt_brats_v4_e1/brats_val_cct_diverse_nearest_smoke4_metrics.json"),
    )
    parser.add_argument(
        "--et-focused-json",
        type=Path,
        default=Path("runs/_ood_adapt_utsw_to_brats_v5_et_focus_e3/epoch_001.json"),
    )
    parser.add_argument(
        "--causal-et-json",
        type=Path,
        default=Path("runs/_ood_causal_adapt_brats_v5_et_focus_e2/epoch_001.json"),
    )
    parser.add_argument(
        "--causal-et-fine-json",
        type=Path,
        default=Path("runs/_ood_causal_adapt_brats_v5_et_focus_e2/brats_val_fine_thresholds_metrics.json"),
    )
    parser.add_argument(
        "--causal-et-precision-json",
        type=Path,
        default=Path("runs/_ood_causal_adapt_brats_v5_et_precision_e2/epoch_001.json"),
    )
    parser.add_argument(
        "--causal-et-structural-json",
        type=Path,
        default=Path("runs/_ood_causal_adapt_brats_v5_et_precision_e2/brats_val_structural_min32_metrics.json"),
    )
    parser.add_argument("--target-mean-dice", type=float, default=0.70)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print(report)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
