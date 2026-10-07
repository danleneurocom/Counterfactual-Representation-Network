"""Choose region thresholds and the ET minimum-component size on the SOURCE validation split.

Never run this on a test split or on the target domain: ``evaluate.py`` refuses calibration
files that were not produced here.

    python -m trace_seg3d.calibrate --ckpt runs/isbi/utsw/trace_s0/best.pt \
        --data-dir data/processed/utsw_128 --splits splits/utsw.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from trace_seg3d.data import CaseDataset, load_splits
from trace_seg3d.evaluate import forward_probs, load_checkpoint
from trace_seg3d.metrics import REGIONS, region_targets_np, structural_prior, threshold_regions

THRESHOLD_GRID = [round(v, 2) for v in np.arange(0.3, 0.81, 0.05)]
MIN_ET_GRID = [0, 8, 16, 32, 64, 128]


def _score(cache: list[dict[str, Any]], thresholds: dict[str, float], min_et: int) -> dict[str, float]:
    """Dice only (fast); HD95 is not used for selection."""

    rows = []
    for item in cache:
        masks = structural_prior(threshold_regions(item["prob"], thresholds), min_et)
        row = {}
        for r in REGIONS:
            p, g = masks[r], item["ref"][r]
            s = float(p.sum() + g.sum())
            row[f"{r}_dice"] = 1.0 if s == 0 else 2.0 * float((p & g).sum()) / s
        row["mean_dice"] = float(np.mean([row[f"{r}_dice"] for r in REGIONS]))
        rows.append(row)
    return {k: float(np.mean([r[k] for r in rows])) for k in ("WT_dice", "TC_dice", "ET_dice", "mean_dice")}


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model, config, _ = load_checkpoint(args.ckpt, device)
    splits = load_splits(args.splits)
    val_ids = splits["val"]
    if sorted(val_ids) != sorted(config["val_ids"]) and not args.allow_split_mismatch:
        raise AssertionError("val split differs from the one used for training this checkpoint")
    ds = CaseDataset(args.data_dir, val_ids)
    names = {m.get("dataset") for m in ds.metas()}
    if names != {config["source_dataset"]}:
        raise AssertionError(f"calibration data {names} is not the source dataset {config['source_dataset']}")
    cache = []
    for batch in DataLoader(ds, batch_size=1, num_workers=args.workers):
        prob = forward_probs(model, batch["image"].to(device), args.amp)[0].cpu().numpy().astype(np.float16).astype(np.float32)
        cache.append({"prob": prob, "ref": region_targets_np(batch["target"][0].numpy() > 0.5), "spacing": float(batch["spacing"][0])})

    base = {"WT": 0.5, "TC": 0.5, "ET": 0.5}
    before = _score(cache, base, 0)
    best = dict(base)
    # coordinate search: each region's threshold maximises that region's Dice
    for r in REGIONS:
        scores = []
        for t in THRESHOLD_GRID:
            trial = {**best, r: t}
            scores.append((_score(cache, trial, 0)[f"{r}_dice"], -abs(t - 0.5), t))
        best[r] = max(scores)[2]
    prior_scores = [(_score(cache, best, m)["mean_dice"], -m, m) for m in MIN_ET_GRID]
    min_et = max(prior_scores)[2]
    after = _score(cache, best, min_et)
    result = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "source_dataset": config["source_dataset"],
        "split": "val",
        "n_cases": len(cache),
        "thresholds": best,
        "min_et_voxels": int(min_et),
        "val_before": before,
        "val_after": after,
        "grid": {"thresholds": THRESHOLD_GRID, "min_et": MIN_ET_GRID},
        "sensitivity": {
            "min_et": {str(m): s for s, _, m in prior_scores},
        },
    }
    out = args.out or Path(args.ckpt).with_name("calib.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--allow-split-mismatch", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return calibrate(parse_args(argv))


if __name__ == "__main__":
    main()
