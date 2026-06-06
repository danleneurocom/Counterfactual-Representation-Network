from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch.optim import AdamW

from baselines.mednext.common import (
    add_common_train_args,
    build_model_from_args,
    checkpoint_config,
    run_eval_epoch,
    run_train_epoch,
    save_json,
    _resolve_device,
)
from baselines.segformer3d.train_brats_h5 import _make_dataset, _make_loader, _volume_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedNeXt on BraTS2020 HDF5 volumes.")
    parser.add_argument("--train-csv", default="data/brats/brats_train.csv")
    parser.add_argument("--val-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    add_common_train_args(parser, default_output_dir="runs/mednext_brats_h5_s_k3", default_volume_size=128)
    parser.add_argument("--limit-train-volumes", type=int)
    parser.add_argument("--limit-val-volumes", type=int)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--slice-col", default="slice")
    parser.add_argument("--h5-image-key", default="image")
    parser.add_argument("--h5-mask-key", default="mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids = _volume_ids(args.train_csv, args.volume_col, args.limit_train_volumes)
    val_ids = _volume_ids(args.val_csv, args.volume_col, args.limit_val_volumes)
    splits = {"train": train_ids, "val": val_ids}
    save_json(splits, output_dir / "splits.json")
    save_json(checkpoint_config(args), output_dir / "config.json")

    train_dataset = _make_dataset(args.train_csv, train_ids, args)
    val_dataset = _make_dataset(args.val_csv, val_ids, args)
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    val_loader = _make_loader(val_dataset, args, shuffle=False)

    device = _resolve_device(args.device)
    model = build_model_from_args(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(model, train_loader, optimizer, device, args, desc="mednext-brats-train")
        val_metrics = run_eval_epoch(model, val_loader, device, args, desc="mednext-brats-val")
        monitor = float(val_metrics.get("brats/mean_dice", float("-inf")))
        log = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        save_json(log, output_dir / f"epoch_{epoch:03d}.json")
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss"),
                "val_loss": val_metrics.get("loss"),
                "val_brats_mean_dice": val_metrics.get("brats/mean_dice"),
            }
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": checkpoint_config(args),
            "splits": splits,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if monitor > best_dice:
            best_dice = monitor
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_brats_mean_dice": best_dice, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
