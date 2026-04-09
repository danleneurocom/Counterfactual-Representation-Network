from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from tqdm import tqdm

from crn.data import make_dataloader
from crn.losses import compute_crn_losses
from crn.models import CounterfactualRepresentationNetwork
from crn.utils import (
    load_yaml,
    move_batch_to_device,
    resolve_device,
    save_json,
    seed_everything,
)


def _mean_logs(logs: list[dict[str, Tensor]]) -> dict[str, float]:
    if not logs:
        return {}
    keys = logs[0].keys()
    return {key: float(torch.stack([entry[key].cpu() for entry in logs]).mean()) for key in keys}


def run_epoch(
    model: CounterfactualRepresentationNetwork,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW | None,
    loss_config: dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    epoch_logs: list[dict[str, Tensor]] = []
    iterator = islice(loader, max_batches) if max_batches is not None else loader
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(iterator, total=total_batches, leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"])
            loss, logs = compute_crn_losses(model, batch, outputs, loss_config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            epoch_logs.append(logs)
    return _mean_logs(epoch_logs)


def build_model(config: dict[str, Any]) -> CounterfactualRepresentationNetwork:
    data_config = config["data"]
    model_config = dict(config["model"])
    model_config.setdefault("in_channels", data_config.get("in_channels", 1))
    model_config.setdefault("image_size", data_config.get("image_size", (128, 128)))
    model_config.setdefault("num_classes", len(data_config.get("label_cols") or []))
    model_config.setdefault("num_seg_classes", 1 if data_config.get("mask_col") else 0)
    return CounterfactualRepresentationNetwork(**model_config)


def train(config: dict[str, Any]) -> None:
    seed_everything(int(config.get("seed", 7)))
    train_config = config["training"]
    data_config = dict(config["data"])
    data_config.update(
        {
            "batch_size": train_config.get("batch_size", 8),
            "num_workers": train_config.get("num_workers", 0),
        }
    )
    device = resolve_device(train_config.get("device", "auto"))
    output_dir = Path(train_config.get("output_dir", "runs/crn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.resolved.json")

    train_loader = make_dataloader(data_config, "train", shuffle=True)
    val_loader = make_dataloader(data_config, "val", shuffle=False)
    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 3e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )

    best_val = float("inf")
    epochs = int(train_config.get("epochs", 1))
    for epoch in range(1, epochs + 1):
        train_logs = run_epoch(
            model,
            train_loader,
            optimizer,
            config.get("loss", {}),
            device,
            train_config.get("max_train_batches"),
        )
        val_logs = run_epoch(
            model,
            val_loader,
            None,
            config.get("loss", {}),
            device,
            train_config.get("max_val_batches"),
        )
        val_loss = val_logs.get("loss/total", float("inf"))

        log_line = {
            "epoch": epoch,
            "train_loss": train_logs.get("loss/total"),
            "val_loss": val_loss,
        }
        print(log_line)
        save_json({"train": train_logs, "val": val_logs}, output_dir / f"epoch_{epoch:03d}.json")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Counterfactual Representation Network.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_yaml(args.config))


if __name__ == "__main__":
    main()
