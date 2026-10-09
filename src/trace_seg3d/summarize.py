"""Aggregate evaluation runs into paper tables: mean +- std over seeds and paired Wilcoxon tests.

Expected layout (what scripts/isbi/run_all.sh produces)::

    runs/isbi/<source>/<method>_s<seed>/eval_<target>_test/{summary.json, per_case.csv}

    python -m trace_seg3d.summarize --root runs/isbi --reference baseline --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from trace_seg3d.evaluate import et_extras

COLUMNS = ("WT_dice", "TC_dice", "ET_dice", "mean_dice", "ET_dice_et_present", "WT_hd95", "TC_hd95", "ET_hd95_et_present", "ET_fp_rate_empty")
# ET_hd95_et_present: ET HD95 (mm) over cases that contain ET; ET_fp_rate_empty: share of ET-empty cases with predicted ET.
# (the plain mean ET HD95 is dominated by the 373 mm penalty of those few empty cases; it stays in table.csv as ET_hd95)


def collect(root: Path, prefix: str) -> dict[tuple[str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for summary_path in sorted(root.glob("*/*/eval_*/summary.json")):
        run_dir = summary_path.parent.parent
        match = re.match(r"(.+)_s(\d+)$", run_dir.name)
        method = match.group(1) if match else run_dir.name
        source = run_dir.parent.name
        target = summary_path.parent.name.removeprefix("eval_")
        summary = json.loads(summary_path.read_text())
        if summary.get("debug_max_cases"):
            print(f"skip debug run {summary_path}")
            continue
        with open(summary_path.parent / "per_case.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        per_case = {r["case_id"]: float(r[f"{prefix}mean_dice"]) for r in rows}
        if f"{prefix}ET_pred_ml" in (rows[0] if rows else {}):  # recompute so older evaluations get the new columns
            summary.update(et_extras(rows, prefix))
        groups[(source, target, method)].append({"summary": summary, "per_case": per_case, "seed": match.group(2) if match else "0"})
    return groups


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("runs/isbi"))
    p.add_argument("--prefix", default="final_", help="final_ | raw_ | cct_final_")
    p.add_argument("--reference", default="baseline", help="method used as reference for the Wilcoxon test")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)
    groups = collect(args.root, args.prefix)
    args.out.mkdir(parents=True, exist_ok=True)
    md = ["| source → target | method | seeds | " + " | ".join(COLUMNS) + " | p (Wilcoxon vs ref) |", "|" + "---|" * (len(COLUMNS) + 4)]
    tex = []
    rows_csv = []
    for (source, target, method), runs in sorted(groups.items()):
        vals = {}
        for c in (*COLUMNS, "ET_hd95", "mean_hd95"):
            arr = np.array([r["summary"].get(f"{args.prefix}{c}", np.nan) for r in runs], dtype=float)
            vals[c] = (float(np.nanmean(arr)), float(np.nanstd(arr)))
        ref_runs = groups.get((source, target, args.reference))
        pval = float("nan")
        if ref_runs and method != args.reference:
            cases = sorted(set.intersection(*[set(r["per_case"]) for r in runs + ref_runs]))
            a = np.array([np.mean([r["per_case"][c] for r in runs]) for c in cases])
            b = np.array([np.mean([r["per_case"][c] for r in ref_runs]) for c in cases])
            if len(cases) > 5 and np.any(a != b):
                pval = float(wilcoxon(a, b).pvalue)
        fmt = lambda c: f"{vals[c][0]:.3f}±{vals[c][1]:.3f}" if ("dice" in c or "rate" in c) else f"{vals[c][0]:.2f}±{vals[c][1]:.2f}"
        md.append(f"| {source} → {target} | {method} | {len(runs)} | " + " | ".join(fmt(c) for c in COLUMNS) + f" | {pval:.2g} |")
        tex.append(f"{source}$\\to${target} & {method} & " + " & ".join(f"{vals[c][0]:.3f}" if "dice" in c else f"{vals[c][0]:.2f}" for c in COLUMNS) + " \\\\")
        rows_csv.append({"source": source, "target": target, "method": method, "seeds": len(runs), **{c: v[0] for c, v in vals.items()}, **{f"{c}_std": v[1] for c, v in vals.items()}, "p_vs_ref": pval})
    (args.out / "table.md").write_text("\n".join(md) + "\n")
    (args.out / "table.tex").write_text("\n".join(tex) + "\n")
    if rows_csv:
        with open(args.out / "table.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]))
            writer.writeheader()
            writer.writerows(rows_csv)
    print("\n".join(md))


if __name__ == "__main__":
    main()
