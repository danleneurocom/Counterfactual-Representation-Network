from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from baselines.mednext.common import (
    add_common_train_args,
    build_model_from_args,
    checkpoint_config,
    checkpoint_monitor,
    initialize_model_from_checkpoint,
    initialize_output_bias_from_loader,
    run_eval_epoch,
    run_train_epoch,
    save_json,
    _resolve_device,
)
from baselines.mednext.dataset_cache import maybe_disk_cache_dataset
from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_utsw import _case_ids, _make_splits


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint(path: str | Path | None) -> dict | None:
    if not path:
        return None
    return torch.load(Path(path).expanduser(), map_location="cpu")


def _monitor_from_epoch_logs(output_dir: Path, args: argparse.Namespace) -> tuple[float, str]:
    best_monitor = float("-inf")
    best_key = "brats/mean_dice"
    for path in sorted(output_dir.glob("epoch_*.json")):
        try:
            record = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        val_metrics = record.get("val")
        if not isinstance(val_metrics, dict):
            continue
        monitor, monitor_key = checkpoint_monitor(val_metrics, args)
        if monitor > best_monitor:
            best_monitor = monitor
            best_key = monitor_key
    return best_monitor, best_key


def _make_loader(root: Path, case_ids: list[str], args: argparse.Namespace, shuffle: bool, cache_name: str) -> DataLoader:
    dataset = UTSWGliomaDataset(
        root=root,
        volume_size=args.volume_size,
        case_ids=case_ids,
        crop_margin=args.crop_margin,
        prefer_manual_seg=args.prefer_manual_seg,
        use_ants_modalities=args.use_ants_modalities,
    )
    dataset = maybe_disk_cache_dataset(
        dataset,
        args.disk_cache_dir,
        namespace=f"utsw_{cache_name}",
        signature={
            "dataset": "utsw",
            "split": cache_name,
            "root": str(root),
            "volume_size": args.volume_size,
            "crop_margin": args.crop_margin,
            "prefer_manual_seg": args.prefer_manual_seg,
            "use_ants_modalities": args.use_ants_modalities,
        },
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
    parser.add_argument("--splits-json", help="Reuse an existing train/val/test split instead of regenerating it.")
    parser.add_argument("--resume-checkpoint", help="Resume model and optimizer state from a MedNeXt baseline checkpoint.")
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
    resume_checkpoint = _load_checkpoint(args.resume_checkpoint)

    if args.splits_json:
        splits = _load_json(Path(args.splits_json))
    elif resume_checkpoint is not None and isinstance(resume_checkpoint.get("splits"), dict):
        splits = resume_checkpoint["splits"]
    else:
        splits = _make_splits(_case_ids(data_root, args.limit_cases), args.seed, args.val_fraction, args.test_fraction)
    save_json(splits, output_dir / "splits.json")
    save_json(checkpoint_config(args), output_dir / "config.json")

    train_loader = _make_loader(data_root, splits["train"], args, shuffle=True, cache_name="train")
    val_loader = _make_loader(data_root, splits["val"], args, shuffle=False, cache_name="val")

    device = _resolve_device(args.device)
    model = build_model_from_args(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        if isinstance(resume_checkpoint.get("optimizer"), dict):
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        save_json(
            {
                "checkpoint": str(Path(args.resume_checkpoint).expanduser()),
                "epoch": int(resume_checkpoint.get("epoch", 0)),
                "start_epoch": start_epoch,
            },
            output_dir / "resume_checkpoint.json",
        )
    else:
        init_report = initialize_model_from_checkpoint(model, args.init_checkpoint)
        if init_report:
            save_json(init_report, output_dir / "init_checkpoint.json")
        bias_init = initialize_output_bias_from_loader(model, train_loader, args)
        if bias_init:
            save_json(bias_init, output_dir / "output_bias_init.json")

    best_monitor, best_monitor_key = _monitor_from_epoch_logs(output_dir, args)
    if start_epoch > args.epochs:
        print({"status": "already_complete", "start_epoch": start_epoch, "epochs": args.epochs, "output_dir": str(output_dir)})
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_train_epoch(model, train_loader, optimizer, device, args, desc="mednext-utsw-train")
        val_metrics = run_eval_epoch(model, val_loader, device, args, desc="mednext-utsw-val")
        monitor, monitor_key = checkpoint_monitor(val_metrics, args)
        log = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        save_json(log, output_dir / f"epoch_{epoch:03d}.json")
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss"),
                "val_loss": val_metrics.get("loss"),
                "val_brats_mean_dice": val_metrics.get("brats/mean_dice"),
                "monitor": monitor,
                "monitor_key": monitor_key,
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
        if monitor > best_monitor:
            best_monitor = monitor
            best_monitor_key = monitor_key
            torch.save(checkpoint, output_dir / "best.pt")

    print({"best_val_monitor": best_monitor, "best_val_monitor_key": best_monitor_key, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
