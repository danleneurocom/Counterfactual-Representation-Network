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


def _aligned(per_seed_rows: list[list[dict[str, str]]], dice_key: str, signals: list[str]):
    """Restrict every seed to the common case ids (same order) -> dice [S,N], signals {name: [S,N]}."""

    common = sorted(set.intersection(*[{r["case_id"] for r in rows} for rows in per_seed_rows]))
    idx = [{r["case_id"]: r for r in rows} for rows in per_seed_rows]
    dice = np.array([[float(m[c][dice_key]) for c in common] for m in idx])
    sig = {s: np.array([[float(m[c][s]) for c in common] for m in idx]) for s in signals}
    return common, dice, sig


METRIC_BETTER = {"spearman": -1.0, "auroc": 1.0, "aurc": -1.0}  # sign that means "better"


def _metrics(sig: np.ndarray, dice: np.ndarray, fail: np.ndarray, cases: np.ndarray) -> dict[str, float]:
    """Seed-averaged metrics of one signal on the case subset ``cases`` (indices, may repeat)."""

    out = {"spearman": [], "auroc": [], "aurc": []}
    for s in range(dice.shape[0]):
        d, x, f = dice[s, cases], sig[s, cases], fail[s, cases]
        out["spearman"].append(spearmanr(x, d).statistic if np.std(x) > 0 and np.std(d) > 0 else np.nan)
        out["auroc"].append(auroc(x, f))
        out["aurc"].append(aurc(x, 1 - d))
    return {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in out.items()}


def bootstrap_audit(per_seed_rows: list[list[dict[str, str]]], dice_key: str, fail_dice: float | None, fail_quantile: float | None,
                    signals: list[str], refs: tuple[str, ...] = ("entropy_mean", "tta_std_mean"), n_boot: int = 2000, seed: int = 0,
                    group_fn=None) -> dict:
    """Point estimates, 95% bootstrap CIs (cases resampled jointly across seeds) and paired
    differences vs reference signals with a one-sided bootstrap p (H0: not better than the reference)."""

    common, dice, sig = _aligned(per_seed_rows, dice_key, signals)
    n = len(common)
    if fail_quantile is not None:
        fail = np.stack([d <= np.quantile(d, fail_quantile) for d in dice])
    else:
        fail = dice < float(fail_dice if fail_dice is not None else 0.5)
    rng = np.random.default_rng(seed)
    if group_fn is None:
        samples = [rng.integers(0, n, n) for _ in range(n_boot)]
    else:  # cluster bootstrap: resample patients, keep all rows (e.g. shifted copies) of each patient
        keys = np.array([group_fn(c) for c in common])
        uniq = np.unique(keys)
        members = {k: np.flatnonzero(keys == k) for k in uniq}
        samples = [np.concatenate([members[k] for k in rng.choice(uniq, len(uniq))]) for _ in range(n_boot)]
    full = np.arange(n)
    boot = {s: [_metrics(sig[s], dice, fail, b) for b in samples] for s in signals}
    res: dict = {"n_cases": n, "n_seeds": int(dice.shape[0]), "n_fail_mean": float(fail.sum(1).mean()), "signals": {},
                 "bootstrap": "cluster (patients)" if group_fn is not None else "cases"}
    for s in signals:
        point = _metrics(sig[s], dice, fail, full)
        entry = {}
        for m in METRIC_BETTER:
            vals = np.array([b[m] for b in boot[s]], dtype=float)
            entry[m] = {"value": point[m], "ci95": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}
        for ref in refs:
            if ref == s or ref not in boot:
                continue
            for m, better in METRIC_BETTER.items():
                diffs = np.array([b[m] - r[m] for b, r in zip(boot[s], boot[ref])], dtype=float)
                diffs = diffs[np.isfinite(diffs)]
                p_not_better = float(np.mean(better * diffs <= 0)) if len(diffs) else float("nan")
                entry[f"delta_{m}_vs_{ref}"] = {"value": point[m] - _metrics(sig[ref], dice, fail, full)[m], "p": p_not_better}
        res["signals"][s] = entry
    return res


def format_bootstrap(res: dict, title: str) -> str:
    lines = [f"{title}  (n = {res['n_cases']} cases x {res['n_seeds']} seed(s); mean n_fail = {res['n_fail_mean']:.1f}; 95% bootstrap CI)", "",
             "| signal | Spearman rho | AUROC fail | AURC | dAUROC vs entropy (p) | dAURC vs entropy (p) | dAUROC vs TTA (p) | dAURC vs TTA (p) |",
             "|---|---|---|---|---|---|---|---|"]
    for s, e in res["signals"].items():
        cell = lambda m: f"{e[m]['value']:.3f} [{e[m]['ci95'][0]:.3f}, {e[m]['ci95'][1]:.3f}]"
        def delta(m: str, ref: str) -> str:
            d = e.get(f"delta_{m}_vs_{ref}")
            return "–" if d is None else f"{d['value']:+.3f} ({d['p']:.3f})"
        lines.append(f"| {s} | {cell('spearman')} | {cell('auroc')} | {cell('aurc')} | {delta('auroc', 'entropy_mean')} | "
                     f"{delta('aurc', 'entropy_mean')} | {delta('auroc', 'tta_std_mean')} | {delta('aurc', 'tta_std_mean')} |")
    lines += ["", "p = one-sided bootstrap probability that the signal is NOT better than the reference (small = better)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, object]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--dice-key", default="final_mean_dice")
    p.add_argument("--fail-dice", type=float, default=0.5)
    p.add_argument("--fail-quantile", type=float, help="use the worst q fraction as failures instead of a Dice cut")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for CIs and paired tests (0 = off)")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    all_rows = [read_rows(c) for c in args.csvs]
    per_seed = [analyse(rows, args.dice_key, args.fail_dice, args.fail_quantile) for rows in all_rows]
    signals = [s for s in SIGNALS if all(s in r for r in per_seed)]
    crit = f"worst {args.fail_quantile:.0%}" if args.fail_quantile else f"{args.dice_key} < {args.fail_dice}"
    lines = [
        f"Audit ({len(args.csvs)} run(s); failure = {crit}"
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
    boot = None
    if args.bootstrap and signals:
        boot = bootstrap_audit(all_rows, args.dice_key, args.fail_dice, args.fail_quantile, signals, n_boot=args.bootstrap)
        lines += ["", format_bootstrap(boot, f"Bootstrap (failure = {crit})")]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        args.out.with_suffix(".json").write_text(json.dumps({"per_seed": per_seed, "summary": table, "bootstrap": boot}, indent=2))
    return table


if __name__ == "__main__":
    main()
