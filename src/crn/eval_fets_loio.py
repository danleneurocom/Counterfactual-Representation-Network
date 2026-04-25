"""FeTS leave-one-institution-out evaluation — Phase C.5.

Consumes the manifest produced by `src/crn/data_fets.py::write_loio_splits` and
a directory of trained checkpoints (one per held-out site) and reports:

* per-site region Dice (WT/TC/ET);
* worst-case site Dice;
* inter-site Dice variance.

These are the clinically meaningful robustness signatures emphasized in Pati
et al. 2022 — mean alone hides the tail of the distribution.

Run:

    PYTHONPATH=src python -m crn.eval_fets_loio \
        --manifest data/fets/loio/loio_manifest.csv \
        --checkpoints-dir runs/fets_loio \
        --config configs/eval_ood_fets_loio.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from crn.evaluate import evaluate
from crn.metrics import site_level_summary
from crn.utils import load_yaml, save_json


def _checkpoint_for_site(checkpoints_dir: Path, site: str) -> Path | None:
    candidates = [
        checkpoints_dir / f"holdout_{site}" / "best.pt",
        checkpoints_dir / site / "best.pt",
        checkpoints_dir / f"{site}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _region_dice_from_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    def _read(key: str) -> float:
        for prefix in ("sweep_best_volume/", "volume/", "region_tuned/"):
            full = prefix + key
            if full in metrics:
                return float(metrics[full])
        return 0.0

    return {
        "WT": _read("brats/WT/dice"),
        "TC": _read("brats/TC/dice"),
        "ET": _read("brats/ET/dice"),
    }


def run(
    manifest_csv: Path,
    checkpoints_dir: Path,
    config_path: Path | None,
    output_path: Path | None = None,
    batch_size: int = 2,
    threshold: float = 0.5,
    device_name: str = "auto",
) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_csv)
    if "site" not in manifest.columns or "test_csv" not in manifest.columns:
        raise ValueError("Manifest must contain 'site' and 'test_csv' columns.")
    base_config = load_yaml(str(config_path)) if config_path else None

    per_site: dict[str, dict[str, float]] = {}
    per_site_full: dict[str, dict[str, Any]] = {}

    for row in manifest.itertuples(index=False):
        site = str(row.site)
        checkpoint = _checkpoint_for_site(checkpoints_dir, site)
        if checkpoint is None:
            print(f"[skip] no checkpoint for site {site} under {checkpoints_dir}")
            continue
        eval_config_path = None
        if base_config is not None:
            patched = dict(base_config)
            data = dict(patched.get("data", {}))
            data["val_csv"] = str(row.test_csv)
            patched["data"] = data
            eval_config_path = checkpoints_dir / f"_tmp_eval_{site}.yaml"
            eval_config_path.write_text(_dump_yaml(patched))
        metrics = evaluate(
            checkpoint_path=str(checkpoint),
            split="val",
            config_path=str(eval_config_path) if eval_config_path else None,
            batch_size=batch_size,
            threshold=threshold,
            device_name=device_name,
        )
        if eval_config_path and eval_config_path.exists():
            eval_config_path.unlink()
        per_site[site] = _region_dice_from_metrics(metrics)
        per_site_full[site] = metrics

    summary = site_level_summary(per_site)
    results = {
        "manifest": str(manifest_csv),
        "checkpoints_dir": str(checkpoints_dir),
        "per_site_region_dice": per_site,
        "summary": summary,
    }
    if output_path is None:
        output_path = checkpoints_dir / "fets_loio_summary.json"
    save_json(results, output_path)
    print(json.dumps(summary, indent=2))
    return results


def _dump_yaml(data: Any) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FeTS leave-one-institution-out evaluation runner.")
    parser.add_argument("--manifest", required=True, help="Path to loio_manifest.csv written by data_fets.py.")
    parser.add_argument("--checkpoints-dir", required=True, help="Directory containing per-site checkpoints.")
    parser.add_argument("--config", help="Optional shared eval config; val_csv is rewritten per site.")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        manifest_csv=Path(args.manifest),
        checkpoints_dir=Path(args.checkpoints_dir),
        config_path=Path(args.config) if args.config else None,
        output_path=Path(args.output) if args.output else None,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
