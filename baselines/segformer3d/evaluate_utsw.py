from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_utsw import _build_model, _dice_loss, _resolve_device
from crn.metrics import brats_region_metrics, brats_volume_metrics_from_probs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _mean(items: list[float]) -> float:
    return float(sum(items) / len(items)) if items else float("nan")


def _average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(set().union(*(item.keys() for item in items)))
    return {
        key: _mean([float(item[key]) for item in items if key in item])
        for key in keys
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    splits = checkpoint.get("splits")
    if splits is None:
        splits_path = Path(args.checkpoint).with_name("splits.json")
        splits = _load_json(splits_path)
    case_ids = splits[args.split]

    data_root = Path(args.data_root or config.get("data_root", "data/brats/PKG - UTSW-Glioma/UTSW-Glioma"))
    volume_size = int(args.volume_size or config.get("volume_size", 64))
    model_size = str(args.model_size or config.get("model_size", "tiny"))
    batch_size = int(args.batch_size or config.get("batch_size", 1))

    dataset = UTSWGliomaDataset(
        root=data_root,
        volume_size=volume_size,
        case_ids=case_ids,
        crop_margin=int(config.get("crop_margin", 8)),
        prefer_manual_seg=bool(config.get("prefer_manual_seg", False)),
        use_ants_modalities=bool(config.get("use_ants_modalities", False)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    device = _resolve_device(args.device)
    model = _build_model(model_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    losses: list[float] = []
    batch_metrics: list[dict[str, float]] = []
    volume_metrics: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"eval:{args.split}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        logits = model(image)
        loss = F.binary_cross_entropy_with_logits(logits, target) + _dice_loss(logits, target)
        losses.append(float(loss.detach().cpu()))

        logits_cpu = logits.detach().cpu()
        target_cpu = target.detach().cpu()
        batch_metrics.append(brats_region_metrics(logits_cpu, target_cpu, threshold=args.threshold))
        probs_cpu = torch.sigmoid(logits_cpu)
        for row in range(probs_cpu.shape[0]):
            probs_volume = probs_cpu[row].permute(1, 0, 2, 3).contiguous()
            target_volume = target_cpu[row].permute(1, 0, 2, 3).contiguous()
            volume_metrics.append(
                brats_volume_metrics_from_probs(
                    probs_volume,
                    target_volume,
                    threshold=args.threshold,
                    channel_names=["ncr_net", "edema", "enhancing_tumor"],
                    region_channels={"WT": [0, 1, 2], "TC": [0, 2], "ET": [2]},
                )
            )
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    metrics = {"loss": _mean(losses), "num_cases": float(len(volume_metrics)), "threshold": float(args.threshold)}
    metrics.update(_average_metric_dicts(batch_metrics))
    metrics.update({f"volume/{key}": value for key, value in _average_metric_dicts(volume_metrics).items()})
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SegFormer3D UTSW checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-root")
    parser.add_argument("--output-json")
    parser.add_argument("--model-size", choices=["tiny", "base"])
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    output_json = Path(args.output_json) if args.output_json else Path(args.checkpoint).with_name(f"{args.split}_metrics.json")
    _save_json(metrics, output_json)
    print(metrics)
    print({"metrics_json": str(output_json)})


if __name__ == "__main__":
    main()
