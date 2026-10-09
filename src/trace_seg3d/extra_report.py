"""Tables for the audit paper from the extra evaluations written by the Kaggle runner.

    runs/<src>/<method>_s<seed>/xeval_<target>_shift-<name>-<sev>/   controlled shifts on the source test set
    runs/<src>/<method>_s<seed>/xeval_<target>_abl-<tag>/             inference-time CCT ablations
    runs/<src>/<method>_s<seed>/probe/probe.json                      causal-assumption probes

Outputs (in --out):
    shift_curves.md / .csv   severity -> Dice and mean audit signal, per method
    shift_audit.md / .json   audit metrics pooled over clean + all shifted cases (failure = worst 20 %),
                             with bootstrap CIs and paired tests vs entropy / TTA
    cct_ablation.md / .csv   K, bank selection and transported levels -> audit quality of CCT signals
    probes.md                which factor predicts context vs disease targets
    localisation.md          voxel AUROC of the audit maps for wrong voxels

    python -m trace_seg3d.extra_report --root runs/isbi --out results/isbi/extra
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from trace_seg3d.audit import SIGNALS, auroc, aurc, bootstrap_audit, format_bootstrap, read_rows

CURVE_SIGNALS = ("cct_disagree", "cct_u_mean", "entropy_mean", "tta_std_mean")
SHIFT_RE = re.compile(r"xeval_(?P<target>.+)_shift-(?P<name>[a-z]+)-(?P<sev>\d)$")
ABL_RE = re.compile(r"xeval_(?P<target>.+)_abl-(?P<tag>[a-z0-9]+)$")


def _run_parts(run_dir: Path) -> tuple[str, str, str]:
    m = re.match(r"(.+)_s(\d+)$", run_dir.name)
    return run_dir.parent.name, (m.group(1) if m else run_dir.name), (m.group(2) if m else "0")


def _mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return float(np.mean(vals)) if vals else float("nan")


def shift_tables(root: Path, out: Path) -> list[str]:
    # (src, method) -> seed -> (shift, severity) -> rows
    groups: dict[tuple[str, str], dict[str, dict[tuple[str, int], list[dict[str, str]]]]] = defaultdict(lambda: defaultdict(dict))
    for d in sorted(root.glob("*/*/xeval_*_shift-*")):
        m = SHIFT_RE.match(d.name)
        if not m or not (d / "per_case.csv").exists():
            continue
        src, method, seed = _run_parts(d.parent)
        per_seed = groups[(src, method)][seed]
        per_seed[(m["name"], int(m["sev"]))] = read_rows(d / "per_case.csv")
        clean = d.parent / f"eval_{m['target']}_test" / "per_case.csv"
        if clean.exists() and ("clean", 0) not in per_seed:
            per_seed[("clean", 0)] = read_rows(clean)
    if not groups:
        return []
    lines = ["# Controlled acquisition shifts (source test set; mean ± std over seeds)", "",
             "| source | method | seeds | shift | severity | mean Dice | " + " | ".join(CURVE_SIGNALS) + " |", "|" + "---|" * (6 + len(CURVE_SIGNALS))]
    csv_rows = []
    audit_lines = ["# Audit pooled over clean + shifted test cases (failure = worst 20 % per seed; cluster bootstrap over patients)", ""]
    audit_json = {}
    for (src, method), seeds in sorted(groups.items()):
        conditions = sorted(set.intersection(*[set(v) for v in seeds.values()]), key=lambda c: (c[0] != "clean", c))
        for cond in conditions:
            per = {k: [_mean(seeds[sd][cond], k) for sd in seeds] for k in ("final_mean_dice", *CURVE_SIGNALS)}
            fmt = lambda k, nd: f"{np.mean(per[k]):.{nd}f}" + (f"±{np.std(per[k]):.{nd}f}" if len(seeds) > 1 else "")
            lines.append(f"| {src} | {method} | {len(seeds)} | {cond[0]} | {cond[1]} | {fmt('final_mean_dice', 3)} | " + " | ".join(fmt(k, 4) for k in CURVE_SIGNALS) + " |")
            csv_rows.append({"source": src, "method": method, "n_seeds": len(seeds), "shift": cond[0], "severity": cond[1],
                             **{f"{k}_mean": float(np.mean(v)) for k, v in per.items()}, **{f"{k}_std": float(np.std(v)) for k, v in per.items()}})
        # pooled audit: each (patient, condition) is one sample; seeds aligned on the same samples
        pooled = [[dict(r, case_id=f"{r['case_id']}|{c[0]}{c[1]}") for c in conditions for r in seeds[sd][c]] for sd in sorted(seeds)]
        signals = [s for s in SIGNALS if all(p and all(r.get(s) not in (None, "") for r in p) for p in pooled)]
        if len(pooled[0]) >= 20 and signals:
            res = bootstrap_audit(pooled, "final_mean_dice", None, 0.2, signals, n_boot=1000, group_fn=lambda c: c.split("|")[0])
            audit_json[f"{src}/{method}"] = res
            audit_lines += [format_bootstrap(res, f"## {src} / {method} ({len(seeds)} seed(s), {len(conditions)} conditions)"), ""]
    (out / "shift_curves.md").write_text("\n".join(lines) + "\n")
    with open(out / "shift_curves.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (out / "shift_audit.md").write_text("\n".join(audit_lines) + "\n")
    (out / "shift_audit.json").write_text(json.dumps(audit_json, indent=2))
    return lines + [""] + audit_lines


def localisation_table(root: Path, out: Path) -> list[str]:
    """Voxel-level: does the audit MAP point at wrong voxels? (mean per-case AUROC; needs the newer evaluate.py)."""

    rows_out = []
    for f in sorted(root.glob("*/*/*eval_*/per_case.csv")):
        rows = read_rows(f)
        if not rows or "loc_auroc_entropy" not in rows[0]:
            continue
        src, method, seed = _run_parts(f.parent.parent)
        cond = f.parent.name
        vals = {k: _mean(rows, k) for k in ("loc_auroc_cct_u", "loc_auroc_entropy", "loc_auroc_tta")}
        rows_out.append((src, method, seed, cond, vals))
    if not rows_out:
        return []
    lines = ["# Failure localisation (voxel AUROC of the audit map for wrong voxels near the tumour; mean over cases)", "",
             "| source | method | seed | evaluation | CCT instability | entropy | TTA std |", "|---|---|---|---|---|---|---|"]
    for src, method, seed, cond, v in rows_out:
        lines.append(f"| {src} | {method} | {seed} | {cond} | {v['loc_auroc_cct_u']:.3f} | {v['loc_auroc_entropy']:.3f} | {v['loc_auroc_tta']:.3f} |")
    (out / "localisation.md").write_text("\n".join(lines) + "\n")
    return lines


def _audit_quality(rows: list[dict[str, str]], signal: str) -> dict[str, float]:
    dice = np.array([float(r["final_mean_dice"]) for r in rows])
    sig = np.array([float(r[signal]) for r in rows])
    fail = dice <= np.quantile(dice, 0.2)
    from scipy.stats import spearmanr

    rho = spearmanr(sig, dice).statistic if np.std(sig) > 0 else float("nan")
    return {"spearman": float(rho), "auroc_q20": auroc(sig, fail), "aurc": aurc(sig, 1 - dice)}


def ablation_tables(root: Path, out: Path) -> list[str]:
    entries = []
    for d in sorted(root.glob("*/*/xeval_*_abl-*")):
        m = ABL_RE.match(d.name)
        if not m or not (d / "per_case.csv").exists():
            continue
        src, method, seed = _run_parts(d.parent)
        entries.append((src, method, seed, m["target"], m["tag"], d))
        default = d.parent / f"eval_{m['target']}_test"
        if (default / "per_case.csv").exists() and not any(e[4] == "default" and e[5] == default for e in entries):
            entries.append((src, method, seed, m["target"], "default", default))
    if not entries:
        return []
    order = {"default": 0, "k1": 1, "k2": 2, "k8": 3, "random": 4, "skips": 5, "bottleneck": 6}
    lines = ["# Inference-time CCT ablation (default = K 4, diverse bank, all levels; failure = worst 20 %)", "",
             "| source → target | method | variant | cct_final mean Dice | cct_disagree rho / AUROC / AURC | cct_u_mean rho / AUROC / AURC |", "|---|---|---|---|---|---|"]
    csv_rows = []
    for src, method, seed, target, tag, d in sorted(entries, key=lambda e: (e[0], e[3], e[1], order.get(e[4], 9))):
        rows = read_rows(d / "per_case.csv")
        q = {s: _audit_quality(rows, s) for s in ("cct_disagree", "cct_u_mean") if rows and rows[0].get(s) not in (None, "")}
        cell = lambda s: " / ".join(f"{q[s][k]:.3f}" for k in ("spearman", "auroc_q20", "aurc")) if s in q else "–"
        dice = _mean(rows, "cct_final_mean_dice")
        lines.append(f"| {src} → {target} | {method} | {tag} | {dice:.3f} | {cell('cct_disagree')} | {cell('cct_u_mean')} |")
        csv_rows.append({"source": src, "target": target, "method": method, "seed": seed, "variant": tag, "cct_final_mean_dice": dice,
                         **{f"{s}_{k}": v for s, qq in q.items() for k, v in qq.items()}})
    (out / "cct_ablation.md").write_text("\n".join(lines) + "\n")
    with open(out / "cct_ablation.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in csv_rows for k in r}))
        writer.writeheader()
        writer.writerows(csv_rows)
    return lines


def probe_tables(root: Path, out: Path) -> list[str]:
    files = sorted(root.glob("*/*/probe/probe.json"))
    if not files:
        return []
    lines = ["# Causal-assumption probes (linear, PCA-matched; train = source train, score = source test)", "",
             "| source | method | target | kind | metric | chance | from moments (z_c, swapped by CCT) | from content (z_d, kept) |", "|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.loads(f.read_text())
        src, method, _ = _run_parts(f.parent.parent)
        for name, p in r["probes"].items():
            lines.append(f"| {src} | {method} | {name} | {p['kind']} | {p['metric']} | {p['chance']:.2f} | {p['from_moments']:.3f} | {p['from_content']:.3f} |")
    lines += ["", "Assumption holds if context targets (site/scanner/domain) are predicted better from z_c and disease targets (grade, volume) better from z_d."]
    (out / "probes.md").write_text("\n".join(lines) + "\n")
    return lines


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("runs/isbi"))
    p.add_argument("--out", type=Path, default=Path("results/isbi/extra"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    text = (shift_tables(args.root, args.out) + [""] + ablation_tables(args.root, args.out) + [""] + probe_tables(args.root, args.out)
            + [""] + localisation_table(args.root, args.out))
    print("\n".join(text))


if __name__ == "__main__":
    main()
