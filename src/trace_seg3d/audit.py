"""Does an audit signal flag the cases where segmentation fails?  (main evidence for the audit claim)

For every signal column in ``per_case.csv`` (cct_u_mean, cct_u_frac, cct_disagree, entropy_mean,
tta_std_mean, ens_std_mean) we report
* Spearman rho between signal and per-case Dice (more negative = better),
* AUROC for detecting failed cases (Dice < --fail-dice, or the worst --fail-quantile),
* AURC of the risk-coverage curve when cases are rejected by decreasing signal
  (risk = 1 - Dice; lower = better) and the oracle AURC for reference.
Several CSVs (seeds) -> mean +- std over seeds.

    python -m trace_seg3d.audit runs/isbi/utsw/trace_s*/eval_brats_test/per_case.csv --out audit_utsw_to_brats.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

SIGNALS = ("cct_u_mean", "cct_u_frac", "cct_disagree", "entropy_mean", "tta_std_mean", "ens_std_mean")


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    pos = positives.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aurc(scores: np.ndarray, risk: np.ndarray) -> float:
    """Area under risk-coverage: keep the most confident (lowest score) cases first."""

    order = np.argsort(scores, kind="stable")
    cum = np.cumsum(risk[order]) / np.arange(1, len(risk) + 1)
    return float(cum.mean())


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def analyse(rows: list[dict[str, str]], dice_key: str, fail_dice: float | None, fail_quantile: float | None) -> dict[str, dict[str, float]]:
    dice = np.array([float(r[dice_key]) for r in rows])
    risk = 1 - dice
    if fail_quantile is not None:
        fail = dice <= np.quantile(dice, fail_quantile)
    else:
        fail = dice < float(fail_dice if fail_dice is not None else 0.5)
    out: dict[str, dict[str, float]] = {"_meta": {"n": float(len(rows)), "n_fail": float(fail.sum()), "oracle_aurc": aurc(risk, risk), "random_aurc": float(risk.mean())}}
    for s in SIGNALS:
        if not rows or s not in rows[0] or rows[0][s] in ("", None):
            continue
        sig = np.array([float(r[s]) for r in rows])
        rho = spearmanr(sig, dice).statistic if np.std(sig) > 0 else float("nan")
        out[s] = {"spearman": float(rho), "auroc": auroc(sig, fail), "aurc": aurc(sig, risk)}
    return out


def main(argv: list[str] | None = None) -> dict[str, object]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--dice-key", default="final_mean_dice")
    p.add_argument("--fail-dice", type=float, default=0.5)
    p.add_argument("--fail-quantile", type=float, help="use the worst q fraction as failures instead of a Dice cut")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    per_seed = [analyse(read_rows(c), args.dice_key, args.fail_dice, args.fail_quantile) for c in args.csvs]
    signals = [s for s in SIGNALS if all(s in r for r in per_seed)]
    lines = [
        f"Audit ({len(args.csvs)} run(s); failure = "
        + (f"worst {args.fail_quantile:.0%}" if args.fail_quantile else f"{args.dice_key} < {args.fail_dice}")
        + f"; mean n_fail = {np.mean([r['_meta']['n_fail'] for r in per_seed]):.1f} / {per_seed[0]['_meta']['n']:.0f})",
        "",
        "| signal | Spearman rho (vs Dice) | AUROC fail | AURC (lower better) |",
        "|---|---|---|---|",
    ]
    table: dict[str, object] = {}
    for s in signals:
        vals = {m: np.array([r[s][m] for r in per_seed], dtype=float) for m in ("spearman", "auroc", "aurc")}
        table[s] = {m: [float(np.nanmean(v)), float(np.nanstd(v))] for m, v in vals.items()}
        lines.append(f"| {s} | " + " | ".join(f"{np.nanmean(vals[m]):.3f} ± {np.nanstd(vals[m]):.3f}" for m in ("spearman", "auroc", "aurc")) + " |")
    lines.append(f"| oracle | – | 1.000 | {np.mean([r['_meta']['oracle_aurc'] for r in per_seed]):.3f} |")
    lines.append(f"| random | 0 | 0.500 | {np.mean([r['_meta']['random_aurc'] for r in per_seed]):.3f} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        args.out.with_suffix(".json").write_text(json.dumps({"per_seed": per_seed, "summary": table}, indent=2))
    return table


if __name__ == "__main__":
    main()
