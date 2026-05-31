from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

import torch

from crn.train import train
from crn.utils import load_yaml


def _is_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda" in message and "memory" in message


def _resolve_wandb_config(args: argparse.Namespace, batch_size: int, config_path: Path) -> dict[str, object]:
    project = args.wandb_project or os.getenv("WANDB_PROJECT")
    if not project:
        return {}
    run_name = args.wandb_name or f"{config_path.stem}-bs{batch_size}"
    return {
        "project": project,
        "entity": args.wandb_entity or os.getenv("WANDB_ENTITY"),
        "name": run_name,
        "tags": ["oom-retry", f"bs{batch_size}"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with automatic OOM batch-size backoff.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--batch-multiplier", type=int, default=64)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    base_config = load_yaml(config_path)
    base_batch = int(base_config.get("training", {}).get("batch_size", 1))
    target_batch = max(1, base_batch * int(args.batch_multiplier))
    min_batch = max(1, int(args.min_batch_size))

    batch_size = target_batch
    attempt = 0
    while batch_size >= min_batch:
        attempt += 1
        config = copy.deepcopy(base_config)
        training = config.setdefault("training", {})
        training["batch_size"] = batch_size

        init_checkpoint = training.get("init_checkpoint")
        if init_checkpoint and not Path(init_checkpoint).exists():
            print(f"init_checkpoint not found: {init_checkpoint}. Running without warmstart.")
            training.pop("init_checkpoint", None)
            training.pop("init_strict", None)

        base_output_dir = Path(training.get("output_dir", "runs/crn"))
        training["output_dir"] = str(base_output_dir) + f"_bs{batch_size}"

        wandb_config = _resolve_wandb_config(args, batch_size, config_path)
        if wandb_config:
            config["wandb"] = wandb_config

        print({"attempt": attempt, "batch_size": batch_size, "output_dir": training["output_dir"]})
        try:
            train(config)
            return
        except RuntimeError as exc:
            if _is_oom(exc):
                print(f"OOM at batch_size={batch_size}; retrying with batch_size={batch_size // 2}")
                batch_size = max(min_batch, batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

    raise RuntimeError("All retries failed due to OOM.")


if __name__ == "__main__":
    main()
