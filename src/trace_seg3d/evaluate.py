"""Per-case evaluation on a test split (in-distribution or OOD) + audit signals.

Guarantees enforced here (they were violated by the old evaluators):
* thresholds / structural prior come from ``calibrate.py`` on the SOURCE validation split of the
  SAME training run (checked);
* the CCT context bank is the source-train bank stored in the checkpoint (never target data);
* no metadata (grade, tumour type, ...) and no ground truth is used to produce predictions.

Prediction variants per case
    raw          thresholds 0.5, no prior
    final        calibrated thresholds + structural prior          <- report this one
    cct_final    CCT consensus probabilities, calibrated + prior   (only with --cct-k > 0)

Audit signals per case (higher = less trustworthy), all computed without labels
    cct_u_mean / cct_u_frac / cct_disagree   CCT instability (requires a bank)
    entropy_mean                             mean binary entropy in predicted WT
    tta_std_mean                             std over mirror flips (--tta)
    ens_std_mean                             std over other seeds (--ensemble-ckpts)

    python -m trace_seg3d.evaluate --ckpt runs/isbi/utsw/trace_s0/best.pt \
        --calib runs/isbi/utsw/trace_s0/calib.json \
        --data-dir data/processed/brats_128 --splits splits/brats.json --split test \
        --cct-k 4 --tta --out runs/isbi/utsw/trace_s0/eval_brats_test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from trace_seg3d.data import CaseDataset, load_splits
from trace_seg3d.metrics import REGIONS, case_metrics, region_targets_np, structural_prior, threshold_regions
from trace_seg3d.model import ContextBank, TraceMedNeXt

DEFAULT_THRESHOLDS = {"WT": 0.5, "TC": 0.5, "ET": 0.5}


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[TraceMedNeXt, dict[str, Any], ContextBank | None]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TraceMedNeXt(**state["config"]["model"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    bank = ContextBank.from_state(state["bank"]) if state.get("bank") else None
    if bank is not None and bank.dataset != state["config"]["source_dataset"]:
        raise AssertionError("context bank does not come from the source dataset")
    return model, state["config"], bank


@torch.no_grad()
def forward_probs(model: TraceMedNeXt, x: torch.Tensor, amp: bool) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=amp and x.device.type == "cuda"):
        logits = model(x)["logits"]
    return torch.sigmoid(logits.float())


@torch.no_grad()
def tta_probs(model: TraceMedNeXt, x: torch.Tensor, amp: bool) -> torch.Tensor:
    outs = [forward_probs(model, x, amp)]
    for axis in (2, 3, 4):
        outs.append(forward_probs(model, x.flip(axis), amp).flip(axis))
    return torch.stack(outs)


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)


def _roi(mask_wt: np.ndarray, brain: np.ndarray) -> np.ndarray:
    return mask_wt if mask_wt.any() else brain


def _mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * float((a & b).sum()) / float(s)


def load_calibration(path: Path | None, config: dict[str, Any], ckpt: Path) -> dict[str, Any]:
    if path is None:
        return {"thresholds": DEFAULT_THRESHOLDS, "min_et_voxels": 0, "calibrated": False}
    calib = json.loads(Path(path).read_text())
    if calib.get("split") != "val" or calib.get("source_dataset") != config["source_dataset"]:
        raise AssertionError(f"calibration must come from the SOURCE val split ({config['source_dataset']}), got {calib.get('source_dataset')}/{calib.get('split')}")
    if Path(calib["ckpt"]).resolve().parent != Path(ckpt).resolve().parent:
        raise AssertionError("calibration file belongs to a different training run")
    calib["calibrated"] = True
    return calib


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else ("cpu" if args.device == "auto" else args.device))
    model, config, bank = load_checkpoint(args.ckpt, device)
    calib = load_calibration(args.calib, config, args.ckpt)
    ensemble = [load_checkpoint(p, device)[0] for p in (args.ensemble_ckpts or [])]
    splits = load_splits(args.splits)
    ids = splits[args.split][: args.max_cases]
    ds = CaseDataset(args.data_dir, ids)
    loader = DataLoader(ds, batch_size=1, num_workers=args.workers)
    target_name = ds.metas()[0].get("dataset", "NA") if len(ds) else "NA"
    setting = "ID" if target_name == config["source_dataset"] else "OOD"
    if setting == "ID" and args.split != "test" and not args.allow_non_test:
        raise AssertionError("in-distribution numbers must come from the test split")
    if args.cct_k > 0 and bank is None:
        raise AssertionError("checkpoint has no source bank; re-run training to the end")
    thresholds, min_et = calib["thresholds"], int(calib["min_et_voxels"])
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    saved = 0
    for batch in loader:
        x = batch["image"].to(device)
        sub = batch["target"][0].numpy() > 0.5
        brain = batch["brain"][0].numpy().astype(bool)
        spacing = float(batch["spacing"][0])
        ref = region_targets_np(sub)
        row: dict[str, Any] = {"case_id": batch["case_id"][0], "dataset": target_name, "setting": setting, "et_present": int(ref["ET"].any())}

        if args.tta:
            stack = tta_probs(model, x, args.amp)
            prob_t = stack[0]
            row["tta_std_mean_raw"] = stack.std(0)[0].cpu().numpy()
        else:
            prob_t = forward_probs(model, x, args.amp)
        prob = prob_t[0].cpu().numpy()
        raw = threshold_regions(prob, DEFAULT_THRESHOLDS)
        final = structural_prior(threshold_regions(prob, thresholds), min_et)
        row.update(case_metrics(raw, ref, spacing, "raw_"))
        row.update(case_metrics(final, ref, spacing, "final_"))
        roi = _roi(final["WT"], brain)
        row["entropy_mean"] = float(_entropy(prob)[:, roi].mean())
        if "tta_std_mean_raw" in row:
            row["tta_std_mean"] = float(row.pop("tta_std_mean_raw")[:, roi].mean())
        if ensemble:
            probs = [prob] + [forward_probs(m, x, args.amp)[0].cpu().numpy() for m in ensemble]
            row["ens_std_mean"] = float(np.std(np.stack(probs), axis=0)[:, roi].mean())
        if args.cct_k > 0 and bank is not None:
            cct = model.cct(x, bank, args.cct_k, selection=args.cct_selection)
            u = cct["instability"][0].cpu().numpy()
            consensus = cct["consensus"][0].cpu().numpy()
            cct_final = structural_prior(threshold_regions(consensus, thresholds), min_et)
            row.update(case_metrics(cct_final, ref, spacing, "cct_final_"))
            row["cct_u_mean"] = float(u[:, roi].mean())
            row["cct_u_frac"] = float((u.max(axis=0)[roi] > args.tau_u).mean())
            transported = [structural_prior(threshold_regions(p[0].cpu().numpy(), thresholds), min_et) for p in cct["transported"]]
            row["cct_disagree"] = float(1 - np.mean([np.mean([_mask_dice(t[r], final[r]) for r in REGIONS]) for t in transported]))
            if saved < args.save_maps:
                np.savez_compressed(args.out / f"maps_{row['case_id']}.npz", image=batch["image"][0].numpy().astype(np.float16), prob=prob.astype(np.float16), consensus=consensus.astype(np.float16), instability=u.astype(np.float16), target=sub.astype(np.uint8))
                saved += 1
        rows.append(row)
        print(f"{row['case_id']}: final mean Dice {row['final_mean_dice']:.3f}")

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("case_id", "dataset", "setting", "et_present"), k))
    with open(args.out / "per_case.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, config, args, calib, setting)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("final_") and k.endswith("dice")}, indent=1))
    return summary


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(rows: list[dict[str, Any]], config: dict[str, Any], args: argparse.Namespace, calib: dict[str, Any], setting: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ckpt": str(args.ckpt),
        "source_dataset": config["source_dataset"],
        "target_split": f"{rows[0]['dataset'] if rows else 'NA'}/{args.split}",
        "setting": setting,
        "mode": config["mode"],
        "n_cases": len(rows),
        "n_et_present": int(sum(r["et_present"] for r in rows)),
        "calibrated": calib.get("calibrated", False),
        "thresholds": calib["thresholds"],
        "min_et_voxels": calib["min_et_voxels"],
        "debug_max_cases": args.max_cases,
        "cct_k": args.cct_k,
    }
    for prefix in ("raw_", "final_", "cct_final_"):
        if not rows or f"{prefix}mean_dice" not in rows[0]:
            continue
        for r in (*REGIONS, "mean"):
            for m in ("dice", "hd95"):
                vals = np.array([row[f"{prefix}{r}_{m}"] for row in rows], dtype=float)
                out[f"{prefix}{r}_{m}"] = float(np.mean(vals))
                out[f"{prefix}{r}_{m}_std"] = float(np.std(vals))
        et_present = np.array([row[f"{prefix}ET_dice"] for row in rows if row["et_present"]], dtype=float)
        out[f"{prefix}ET_dice_et_present"] = float(et_present.mean()) if len(et_present) else float("nan")
        lo, hi = bootstrap_ci(np.array([row[f"{prefix}mean_dice"] for row in rows], dtype=float))
        out[f"{prefix}mean_dice_ci95"] = [lo, hi]
        for m in ("ET_precision", "ET_recall"):
            vals = np.array([row[f"{prefix}{m}"] for row in rows], dtype=float)
            out[f"{prefix}{m}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        out[f"{prefix}ET_fp_components"] = float(np.mean([row[f"{prefix}ET_fp_components"] for row in rows]))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--calib", type=Path, help="calib.json from calibrate.py (omit = uncalibrated 0.5)")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cct-k", type=int, default=4)
    p.add_argument("--cct-selection", choices=["diverse", "random"], default="diverse")
    p.add_argument("--tau-u", type=float, default=0.05)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--ensemble-ckpts", nargs="*", type=Path)
    p.add_argument("--save-maps", type=int, default=0)
    p.add_argument("--max-cases", type=int, help="debug only -- never for reported numbers")
    p.add_argument("--allow-non-test", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
