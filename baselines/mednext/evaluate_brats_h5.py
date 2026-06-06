from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mednext.common import load_model_for_eval, main_logits, save_json, segmentation_loss, _resolve_device
from baselines.segformer3d.evaluate_causal_brats_h5 import BraTSH5VolumeDataset, _volume_metrics
from baselines.segformer3d.train_utsw import _average_metric_dicts, _mean
from crn.metrics import brats_region_metrics


def _make_loader(dataset: BraTSH5VolumeDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    volume_size = int(args.volume_size or config.get("volume_size", 128))
    batch_size = int(args.batch_size or config.get("batch_size", 1))

    dataset = BraTSH5VolumeDataset(
        csv_path=args.brats_csv,
        data_root=args.data_root or config.get("data_root"),
        volume_size=volume_size,
        limit_volumes=args.max_volumes,
        path_col=args.path_col,
        volume_col=args.volume_col,
        slice_col=args.slice_col,
        image_key=args.h5_image_key,
        mask_key=args.h5_mask_key,
        crop_margin=int(args.crop_margin if args.crop_margin is not None else config.get("crop_margin", 8)),
    )
    loader = _make_loader(dataset, batch_size=batch_size, num_workers=args.num_workers)

    device = _resolve_device(args.device)
    model = load_model_for_eval(checkpoint, args).to(device)
    model.eval()

    losses: list[float] = []
    batch_metrics: list[dict[str, float]] = []
    volume_metrics: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"mednext-brats-eval:{args.split_name}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        output = model(image)
        logits = main_logits(output)
        losses.append(float(segmentation_loss(output, target, config.get("deep_supervision_weights")).detach().cpu()))
        logits_cpu = logits.detach().cpu()
        target_cpu = target.detach().cpu()
        batch_metrics.append(brats_region_metrics(logits_cpu, target_cpu, threshold=args.threshold))
        volume_metrics.extend(_volume_metrics(logits, target, args.threshold))
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    metrics: dict[str, Any] = {
        "method": "MedNeXt on BraTS H5",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split_name,
        "brats_csv": str(args.brats_csv),
        "threshold": float(args.threshold),
        "num_cases": float(len(volume_metrics)),
        "loss": _mean(losses),
    }
    metrics.update(_average_metric_dicts(batch_metrics))
    metrics.update({f"volume/{key}": value for key, value in _average_metric_dicts(volume_metrics).items()})
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MedNeXt checkpoint on BraTS2020 HDF5 volumes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--brats-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    parser.add_argument("--output-json")
    parser.add_argument("--split-name", default="brats_val")
    parser.add_argument("--model-id", choices=["S", "B", "M", "L"])
    parser.add_argument("--kernel-size", type=int, choices=[3, 5])
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-volumes", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-margin", type=int)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--slice-col", default="slice")
    parser.add_argument("--h5-image-key", default="image")
    parser.add_argument("--h5-mask-key", default="mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path(args.checkpoint).with_name(f"{args.split_name}_metrics.json")
    )
    save_json(metrics, output_json)
    print(metrics)
    print({"metrics_json": str(output_json)})


if __name__ == "__main__":
    main()
