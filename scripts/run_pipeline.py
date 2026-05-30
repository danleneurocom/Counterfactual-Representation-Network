#!/usr/bin/env python3
"""Run full CRN pipeline: train then evaluate.

Usage:
    python scripts/run_pipeline.py --config configs/crn_brats_smoke_mednext3d.yaml
    python scripts/run_pipeline.py --config configs/crn_brats_smoke_2d.yaml

The backbone is controlled via the config file (model.backbone_mode).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print("=" * 60)
    print("Running:", " ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full CRN train+eval pipeline.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    parser.add_argument("--checkpoint", default="last", choices=["last", "best"], help="Which checkpoint to evaluate.")
    parser.add_argument("--device", default="auto", help="Device override.")
    parser.add_argument("--wandb", default="offline", choices=["online", "offline", "disabled"], help="W&B mode.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Train
    env = {
        "WANDB_MODE": args.wandb if args.wandb != "disabled" else "",
        "WANDB_DISABLED": "true" if args.wandb == "disabled" else "false",
    }
    train_cmd = [
        sys.executable, "-m", "crn.train",
        "--config", str(config_path),
    ]
    print("[PIPELINE] Step 1/2: Training")
    result = subprocess.run(train_cmd, check=False, env={**os.environ, **env})
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Derive output dir from config if possible, otherwise use default pattern
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg.get("training", {}).get("output_dir", "runs/crn")).resolve()
    checkpoint_path = output_dir / f"{args.checkpoint}.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    eval_cmd = [
        sys.executable, "-m", "crn.evaluate",
        "--checkpoint", str(checkpoint_path),
        "--split", "val",
        "--device", args.device,
    ]
    print("\n[PIPELINE] Step 2/2: Evaluation")
    result = subprocess.run(eval_cmd, check=False, env={**os.environ, **env})
    if result.returncode != 0:
        sys.exit(result.returncode)

    print("\n[PIPELINE] Done!")


if __name__ == "__main__":
    main()
