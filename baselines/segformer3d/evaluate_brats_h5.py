from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.segformer3d.evaluate_causal_brats_h5 import BraTSH5VolumeDataset, _save_json, _volume_metrics
from baselines.segformer3d.train_utsw import _average_metric_dicts, _build_model, _mean, _resolve_device, _segmentation_loss
from crn.metrics import brats_region_metrics


def _make_loader(dataset: BraTSH5VolumeDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    volume_size = int(args.volume_size or config.get("volume_size", 128))
    model_size = str(args.model_size or config.get("model_size", "base"))
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
    model = _build_model(model_size).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    losses: list[float] = []
    batch_metrics: list[dict[str, float]] = []
    volume_metrics: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"brats-h5-eval:{args.split_name}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        logits = model(image)
        if not isinstance(logits, Tensor):
            raise TypeError("SegFormer3D must return a tensor of logits.")
        losses.append(float(_segmentation_loss(logits, target).detach().cpu()))
        batch_metrics.append(brats_region_metrics(logits.detach().cpu(), target.detach().cpu(), threshold=args.threshold))
        volume_metrics.extend(_volume_metrics(logits, target, args.threshold))
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    metrics: dict[str, Any] = {
        "method": "SegFormer3D on BraTS H5",
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
    parser = argparse.ArgumentParser(description="Evaluate a SegFormer3D checkpoint on BraTS2020 HDF5 volumes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--brats-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    parser.add_argument("--output-json")
    parser.add_argument("--split-name", default="brats_val")
    parser.add_argument("--model-size", choices=["tiny", "base"])
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
    _save_json(metrics, output_json)
    print(metrics)
    print({"metrics_json": str(output_json)})


if __name__ == "__main__":
    main()
