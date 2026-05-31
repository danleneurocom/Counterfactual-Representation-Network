from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_env(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Env file not found: {path}")
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
    return env


def _resolve_wandb_key(env: dict[str, str]) -> str | None:
    for key in ("WANDB_API_KEY", "wandb_api_key", "wandb_key", "api_key"):
        value = env.get(key)
        if value:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data conversion + training with OOM backoff and W&B logging.")
    parser.add_argument("--config", default="configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml")
    parser.add_argument("--data-root", default="data/brats20")
    parser.add_argument("--output-dir", default="data/brats")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-multiplier", type=int, default=64)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_path = Path("/workspace/.env")
    env = _parse_env(env_path)
    wandb_key = _resolve_wandb_key(env)
    if not wandb_key:
        raise SystemExit("WANDB API key not found in /workspace/.env")

    os.environ["WANDB_API_KEY"] = wandb_key
    if env.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = env["WANDB_PROJECT"]
    if env.get("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = env["WANDB_ENTITY"]

    train_csv = repo_root / "data" / "brats" / "brats_train.csv"
    val_csv = repo_root / "data" / "brats" / "brats_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_brats20_nifti.py",
                "--data-root",
                args.data_root,
                "--output-dir",
                args.output_dir,
                "--val-fraction",
                str(args.val_fraction),
            ],
            cwd=repo_root,
            check=True,
        )

    cmd = [
        sys.executable,
        "scripts/run_train_with_oom_retry.py",
        "--config",
        args.config,
        "--batch-multiplier",
        str(args.batch_multiplier),
    ]
    project = env.get("WANDB_PROJECT")
    if project:
        cmd.extend(["--wandb-project", project])
    entity = env.get("WANDB_ENTITY")
    if entity:
        cmd.extend(["--wandb-entity", entity])
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(cmd, cwd=repo_root, check=True, env=env)


if __name__ == "__main__":
    main()
