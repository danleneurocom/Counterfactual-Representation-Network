"""Fixed patient-level train/val/test splits, stratified by ET presence (and grade when known).

    python -m trace_seg3d.splits --index data/processed/utsw_128/index.csv --out splits/utsw.json
    python -m trace_seg3d.splits --index data/processed/brats_128/index.csv --out splits/brats.json

Use ``--exclude-auto-labels`` to keep cases whose label is the automatic FeTS output (no manual
correction) out of val/test (they stay in train).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def make_splits(
    rows: list[dict[str, str]],
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 2027,
    exclude_auto_labels: bool = False,
) -> dict[str, list[str]]:
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    rng = random.Random(seed)
    forced_train = []
    strata: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if exclude_auto_labels and row.get("label_source", "manual") != "manual":
            forced_train.append(row["case_id"])
            continue
        key = f"et{row.get('et_present', 'NA')}|{row.get('grade', 'NA')}"
        strata[key].append(row["case_id"])
    out: dict[str, list[str]] = {"train": list(forced_train), "val": [], "test": []}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n = len(ids)
        n_test = round(n * fractions[2])
        n_val = round(n * fractions[1])
        out["test"] += ids[:n_test]
        out["val"] += ids[n_test : n_test + n_val]
        out["train"] += ids[n_test + n_val :]
    for name in out:
        out[name] = sorted(out[name])
    all_ids = out["train"] + out["val"] + out["test"]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("split overlap")
    return out


def read_index(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fractions", default="0.7,0.1,0.2")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--exclude-auto-labels", action="store_true")
    args = parser.parse_args(argv)
    fractions = tuple(float(v) for v in args.fractions.split(","))
    splits = make_splits(read_index(args.index), fractions, args.seed, args.exclude_auto_labels)  # type: ignore[arg-type]
    splits["_meta"] = {"index": str(args.index), "fractions": list(fractions), "seed": args.seed}  # type: ignore[assignment]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2))
    print({k: len(v) for k, v in splits.items() if k != "_meta"}, "->", args.out)


if __name__ == "__main__":
    main()
