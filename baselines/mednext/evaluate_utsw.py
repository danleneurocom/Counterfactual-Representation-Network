from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.mednext.calibration import (
    CALIBRATION_OBJECTIVES,
    BratsRegionThresholdSweep,
    brats_region_metrics_from_thresholds,
    parse_region_thresholds,
    parse_threshold_candidates,
    prefix_metrics,
)
from baselines.mednext.common import (
    load_model_for_eval,
    main_logits,
    mirror_tta_logits,
    parse_mirror_tta_axes,
    save_json,
    segmentation_loss,
    _resolve_device,
)
from baselines.segformer3d.data import UTSWGliomaDataset
from baselines.segformer3d.train_utsw import _average_metric_dicts, _mean
from baselines.segformer3d.train_causal_utsw import _prefix_metrics
from crn.metrics import brats_region_metrics, brats_structural_region_metrics, brats_volume_metrics_from_probs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_ids_from_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    case_ids = [item.strip() for item in value.split(",") if item.strip()]
    return case_ids or None


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    splits = checkpoint.get("splits") or _load_json(Path(args.checkpoint).with_name("splits.json"))
    case_ids = _case_ids_from_arg(args.case_ids) or splits[args.split]

    data_root = Path(args.data_root or config.get("data_root", "data/brats/PKG - UTSW-Glioma/UTSW-Glioma"))
    volume_size = int(args.volume_size or config.get("volume_size", 64))
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
    model = load_model_for_eval(checkpoint, args).to(device)
    model.eval()
    region_thresholds = parse_region_thresholds(args.region_thresholds)
    calibration_candidates = parse_threshold_candidates(args.calibration_thresholds)
    calibration_sweep = (
        BratsRegionThresholdSweep(calibration_candidates, objective=args.calibration_objective)
        if calibration_candidates
        else None
    )
    mirror_axes = parse_mirror_tta_axes(args.mirror_tta_axes)

    losses: list[float] = []
    batch_metrics: list[dict[str, float]] = []
    region_calibrated_metrics: list[dict[str, float]] = []
    structural_batch_metrics: list[dict[str, float]] = []
    volume_metrics: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"mednext-utsw-eval:{args.split}", leave=False), start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        output = model(image)
        logits = main_logits(output)
        losses.append(float(segmentation_loss(output, target, config.get("deep_supervision_weights")).detach().cpu()))
        logits = mirror_tta_logits(lambda augmented: main_logits(model(augmented)), image, mirror_axes, base_logits=logits)
        logits_cpu = logits.detach().cpu()
        target_cpu = target.detach().cpu()
        batch_metrics.append(brats_region_metrics(logits_cpu, target_cpu, threshold=args.threshold))
        if region_thresholds is not None:
            region_calibrated_metrics.append(brats_region_metrics_from_thresholds(logits_cpu, target_cpu, region_thresholds))
        if calibration_sweep is not None:
            calibration_sweep.update(logits_cpu, target_cpu)
        if args.structural_prior:
            structural_batch_metrics.append(
                brats_structural_region_metrics(
                    logits_cpu,
                    target_cpu,
                    threshold=args.structural_threshold,
                    min_component_size=args.structural_min_component_size,
                    fill_holes=args.structural_fill_holes,
                    keep_largest=args.structural_keep_largest,
                )
            )
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

    metrics: dict[str, Any] = {
        "method": "MedNeXt on UTSW",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "threshold": float(args.threshold),
        "mirror_tta_axes": ",".join(str(axis) for axis in mirror_axes),
        "num_cases": float(len(volume_metrics)),
        "loss": _mean(losses),
    }
    metrics.update(_average_metric_dicts(batch_metrics))
    if region_calibrated_metrics:
        metrics.update(prefix_metrics(_average_metric_dicts(region_calibrated_metrics), "region_calibrated"))
    if calibration_sweep is not None:
        metrics.update(prefix_metrics(calibration_sweep.summary(), "sweep_region_calibrated"))
    if structural_batch_metrics:
        metrics.update(_prefix_metrics(_average_metric_dicts(structural_batch_metrics), "structural"))
    metrics.update({f"volume/{key}": value for key, value in _average_metric_dicts(volume_metrics).items()})
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MedNeXt UTSW checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--case-ids", help="Comma-separated case ids to evaluate instead of the whole split.")
    parser.add_argument("--data-root")
    parser.add_argument("--output-json")
    parser.add_argument("--model-id", choices=["S", "B", "M", "L"])
    parser.add_argument("--kernel-size", type=int, choices=[3, 5])
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--region-thresholds", help="Fixed WT/TC/ET thresholds, e.g. 'WT=0.4,TC=0.5,ET=0.6'.")
    parser.add_argument("--calibration-thresholds", help="Comma-separated WT/TC/ET threshold grid for validation-time sweep.")
    parser.add_argument(
        "--calibration-objective",
        choices=CALIBRATION_OBJECTIVES,
        default="mean",
        help="Objective used when choosing thresholds from --calibration-thresholds.",
    )
    parser.add_argument("--mirror-tta-axes", help="Optional spatial mirror TTA axes: d,h,w or z,y,x.")
    parser.add_argument("--structural-prior", action="store_true")
    parser.add_argument("--structural-threshold", type=float, default=0.1)
    parser.add_argument("--structural-min-component-size", type=int, default=16)
    parser.add_argument("--structural-fill-holes", action="store_true")
    parser.add_argument("--structural-keep-largest", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    output_json = Path(args.output_json) if args.output_json else Path(args.checkpoint).with_name(f"{args.split}_metrics.json")
    save_json(metrics, output_json)
    print(metrics)
    print({"metrics_json": str(output_json)})


if __name__ == "__main__":
    main()
