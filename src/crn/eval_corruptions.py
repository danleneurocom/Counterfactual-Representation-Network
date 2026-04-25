"""ROOD-MRI / TorchIO corruption OOD evaluation — Phase C.4.

Implements the ROOD-MRI protocol of Boone et al. 2023 (NeuroImage
DOI 10.1016/j.neuroimage.2023.120289; arXiv:2203.06060) using TorchIO
(Pérez-García et al. 2021, arXiv:2003.04696). For each (transform, severity)
pair we apply the corruption to the validation split, run inference, record
region-wise Dice, and aggregate into a mean Corruption Error (mCE) alongside
Dice-vs-severity curves — the figure that empirically validates the
counterfactual-invariance claim.

Run:

    PYTHONPATH=src python -m crn.eval_corruptions \
        --checkpoint runs/brats_segonly_unet_causal_contrastive_mednextL/best.pt \
        --config configs/eval_ood_corruptions.yaml
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from crn.data import ImagingCsvDataset, make_dataloader
from crn.evaluate import VolumeMetricTracker, _load_config
from crn.metrics import mean_corruption_error
from crn.train import build_model
from crn.utils import load_yaml, move_batch_to_device, resolve_device, save_json


CORRUPTION_PRESETS: dict[str, dict[str, list[float]]] = {
    "bias_field": {"coefficients": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "motion": {"num_transforms": [1, 2, 3, 4, 5]},
    "ghosting": {"num_ghosts": [2, 4, 6, 8, 10]},
    "noise": {"std": [0.01, 0.025, 0.05, 0.075, 0.1]},
    "gamma": {"log_gamma": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "spike": {"num_spikes": [1, 2, 3, 4, 5]},
    "blur": {"std": [0.5, 1.0, 1.5, 2.0, 2.5]},
}


def _try_import_torchio():
    try:
        import torchio as tio
        return tio
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "TorchIO is required for Phase C.4 corruption evaluation. Install with `pip install torchio`."
        ) from exc


def _build_transform(name: str, severity: int):
    tio = _try_import_torchio()
    if name not in CORRUPTION_PRESETS:
        raise ValueError(f"Unknown corruption preset {name!r}. Choose from {list(CORRUPTION_PRESETS)}.")
    preset = CORRUPTION_PRESETS[name]
    if name == "bias_field":
        value = float(preset["coefficients"][severity - 1])
        return tio.RandomBiasField(coefficients=(value, value))
    if name == "motion":
        value = int(preset["num_transforms"][severity - 1])
        return tio.RandomMotion(num_transforms=value, image_interpolation="linear")
    if name == "ghosting":
        value = int(preset["num_ghosts"][severity - 1])
        return tio.RandomGhosting(num_ghosts=(value, value))
    if name == "noise":
        value = float(preset["std"][severity - 1])
        return tio.RandomNoise(std=(value, value))
    if name == "gamma":
        value = float(preset["log_gamma"][severity - 1])
        return tio.RandomGamma(log_gamma=(-value, value))
    if name == "spike":
        value = int(preset["num_spikes"][severity - 1])
        return tio.RandomSpike(num_spikes=(value, value))
    if name == "blur":
        value = float(preset["std"][severity - 1])
        return tio.RandomBlur(std=(value, value))
    raise ValueError(f"Unhandled corruption name {name!r}")


def _apply_transform(images: torch.Tensor, transform) -> torch.Tensor:
    """Apply a TorchIO transform to a [B, C, (D,) H, W] tensor."""
    tio = _try_import_torchio()
    if images.ndim == 5:
        corrupted: list[torch.Tensor] = []
        for sample in images:
            subject = tio.Subject(image=tio.ScalarImage(tensor=sample.cpu()))
            out = transform(subject)
            corrupted.append(out["image"].data.float())
        return torch.stack(corrupted, dim=0).to(images.device)
    if images.ndim == 4:
        corrupted_slices: list[torch.Tensor] = []
        for sample in images:
            volume = sample.unsqueeze(-1).cpu()  # (C, H, W, 1)
            subject = tio.Subject(image=tio.ScalarImage(tensor=volume))
            out = transform(subject)
            corrupted_slices.append(out["image"].data.squeeze(-1).float())
        return torch.stack(corrupted_slices, dim=0).to(images.device)
    raise ValueError(f"Unsupported image tensor shape {tuple(images.shape)}")


def _region_dice(volume_summary: dict[str, float]) -> dict[str, float]:
    regions = ("WT", "TC", "ET")
    return {r: float(volume_summary.get(f"brats/{r}/dice", 0.0)) for r in regions}


def _evaluate_once(
    model: torch.nn.Module,
    loader,
    data_config: dict,
    device: torch.device,
    transform=None,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, float]:
    tracker = VolumeMetricTracker([threshold], data_config)
    iterator = islice(loader, max_batches) if max_batches is not None else loader
    with torch.no_grad():
        for batch in tqdm(iterator, leave=False):
            batch_device = move_batch_to_device(batch, device)
            images = batch_device["image"]
            if transform is not None:
                images = _apply_transform(images, transform)
            outputs = model({**batch_device, "image": images}["image"])
            seg_logits = outputs["seg_logits"].detach().cpu()
            mask = batch_device["mask"].detach().cpu()
            tracker.add_batch(seg_logits, mask, batch.get("volume"), batch.get("slice"))
    summaries, _, _ = tracker.finalize()
    return _region_dice(summaries.get(threshold, {}))


def run(
    checkpoint_path: str | Path,
    config_path: str | None = None,
    split: str = "val",
    batch_size: int = 2,
    max_batches: int | None = None,
    threshold: float = 0.5,
    transforms: list[str] | None = None,
    severities: list[int] | None = None,
    device_name: str = "auto",
    output_path: str | Path | None = None,
    seed: int = 7,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = _load_config(checkpoint, config_path)
    data_config = dict(config["data"])
    train_config = dict(config.get("training", {}))
    data_config["batch_size"] = batch_size or train_config.get("batch_size", 2)
    data_config["num_workers"] = train_config.get("num_workers", 0)
    device = resolve_device(device_name if device_name != "auto" else train_config.get("device", "auto"))
    loader = make_dataloader(data_config, split, shuffle=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    transforms = transforms or list(CORRUPTION_PRESETS)
    severities = severities or [1, 2, 3, 4, 5]

    source = _evaluate_once(model, loader, data_config, device, None, threshold, max_batches)

    corrupted: dict[tuple[str, int], dict[str, float]] = {}
    for name in transforms:
        for severity in severities:
            transform = _build_transform(name, severity)
            corrupted[(name, severity)] = _evaluate_once(
                model, loader, data_config, device, transform, threshold, max_batches
            )

    mce = mean_corruption_error(source, corrupted)

    # Dice-vs-severity curves: one per (transform, region).
    curves: dict[str, dict[str, list[float]]] = {}
    for name in transforms:
        curves[name] = {"WT": [], "TC": [], "ET": [], "severity": list(severities)}
        for severity in severities:
            entry = corrupted.get((name, severity), {})
            for region in ("WT", "TC", "ET"):
                curves[name][region].append(float(entry.get(region, 0.0)))

    results = {
        "source_region_dice": source,
        "corrupted_region_dice": {f"{name}@{severity}": scores for (name, severity), scores in corrupted.items()},
        "mCE": mce,
        "dice_curves": curves,
        "config_path": str(config_path) if config_path else None,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "threshold": threshold,
    }
    if output_path is None:
        checkpoint_path_obj = Path(checkpoint_path)
        output_path = checkpoint_path_obj.with_name(f"{checkpoint_path_obj.stem}_{split}_corruptions.json")
    save_json(results, output_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROOD-MRI / TorchIO corruption robustness evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional config path.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--transforms",
        nargs="*",
        choices=list(CORRUPTION_PRESETS),
        help="Subset of corruption presets to evaluate (default: all).",
    )
    parser.add_argument("--severities", type=int, nargs="*", help="Severity levels 1-5 (default: 1 2 3 4 5).")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        threshold=args.threshold,
        transforms=args.transforms,
        severities=args.severities,
        device_name=args.device,
        output_path=args.output,
        seed=args.seed,
    )
    print(json.dumps({"mCE": results["mCE"], "source": results["source_region_dice"]}, indent=2))


if __name__ == "__main__":
    main()
