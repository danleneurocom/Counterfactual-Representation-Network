from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from baselines.mednext.common import (
    add_common_train_args,
    build_model_from_args,
    checkpoint_config,
    run_eval_epoch,
    run_train_epoch,
    save_json,
    _resolve_device,
)
from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_utsw import _case_ids, _make_splits


def _make_loader(root: Path, case_ids: list[str], args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = UTSWGliomaDataset(
        root=root,
        volume_size=args.volume_size,
        case_ids=case_ids,
        crop_margin=args.crop_margin,
        prefer_manual_seg=args.prefer_manual_seg,
        use_ants_modalities=args.use_ants_modalities,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedNeXt on UTSW glioma volumes.")
    parser.add_argument("--data-root", default="data/brats/PKG - UTSW-Glioma/UTSW-Glioma")
    add_common_train_args(parser, default_output_dir="runs/mednext_utsw_s_k3", default_volume_size=64)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--prefer-manual-seg", action="store_true")
    parser.add_argument("--use-ants-modalities", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _make_splits(_case_ids(data_root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)
    save_json(splits, output_dir / "splits.json")
    save_json(checkpoint_config(args), output_dir / "config.json")

    train_loader = _make_loader(data_root, splits["train"], args, shuffle=True)
    val_loader = _make_loader(data_root, splits["val"], args, shuffle=False)

    device = _resolve_device(args.device)
    model = build_model_from_args(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(model, train_loader, optimizer, device, args, desc="mednext-utsw-train")
        val_metrics = run_eval_epoch(model, val_loader, device, args, desc="mednext-utsw-val")
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
