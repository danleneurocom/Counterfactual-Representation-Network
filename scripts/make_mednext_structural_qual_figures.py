from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from torch import Tensor
from torch.utils.data import DataLoader

from baselines.mednext.calibration import parse_region_thresholds
from baselines.mednext.evaluate_causal_brats_h5 import _build_model
from baselines.segformer3d.evaluate_causal_brats_h5 import BraTSH5VolumeDataset
from baselines.segformer3d.train_causal_utsw import build_context_bank
from baselines.segformer3d.train_utsw import _resolve_device
from crn.metrics import binary_volume_metrics_from_masks, postprocess_binary_volume


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=None,
        kernel_size=None,
        latent_dim=None,
        base_channels=None,
        region_volume_scale=None,
        et_volume_veto_scale=None,
        et_volume_veto_multiplier=None,
        et_volume_veto_min_fraction=None,
        et_volume_veto_max_bias=None,
    )


def _make_dataset(args: argparse.Namespace, *, csv_path: str | Path, max_volumes: int | None) -> BraTSH5VolumeDataset:
    return BraTSH5VolumeDataset(
        csv_path=csv_path,
        data_root=args.data_root,
        volume_size=args.volume_size,
        limit_volumes=max_volumes,
        path_col=args.path_col,
        volume_col=args.volume_col,
        slice_col=args.slice_col,
        image_key=args.h5_image_key,
        mask_key=args.h5_mask_key,
        crop_margin=args.crop_margin,
    )


def _region_masks_from_probs(sub_probs: np.ndarray, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    return {
        "WT": np.logical_or.reduce(
            [
                sub_probs[0] >= thresholds["WT"],
                sub_probs[1] >= thresholds["WT"],
                sub_probs[2] >= thresholds["WT"],
            ]
        ),
        "TC": np.logical_or(sub_probs[0] >= thresholds["TC"], sub_probs[2] >= thresholds["TC"]),
        "ET": sub_probs[2] >= thresholds["ET"],
    }


def _target_region_masks(target_sub: np.ndarray) -> dict[str, np.ndarray]:
    ncr_net = target_sub[0] > 0.5
    edema = target_sub[1] > 0.5
    et = target_sub[2] > 0.5
    tc = np.logical_or(ncr_net, et)
    wt = np.logical_or.reduce([ncr_net, edema, et])
    return {"WT": wt, "TC": tc, "ET": et}


def _enforce_hierarchy(regions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    et = np.asarray(regions["ET"], dtype=bool)
    tc = np.logical_or(np.asarray(regions["TC"], dtype=bool), et)
    wt = np.logical_or(np.asarray(regions["WT"], dtype=bool), tc)
    return {"WT": wt, "TC": tc, "ET": et}


def _structural_masks(
    regions: dict[str, np.ndarray],
    *,
    min_component_size: int,
    fill_holes: bool,
    keep_largest: bool,
) -> dict[str, np.ndarray]:
    processed = {
        name: postprocess_binary_volume(
            mask,
            min_component_size=min_component_size,
            fill_holes=fill_holes,
            keep_largest=keep_largest,
            connectivity=1,
        )
        for name, mask in regions.items()
    }
    return _enforce_hierarchy(processed)


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    structure = generate_binary_structure(mask.ndim, 1)
    return np.logical_xor(mask, binary_erosion(mask, structure=structure, border_value=0))


def _normalize_slice(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _gray_rgb(image: np.ndarray) -> np.ndarray:
    gray = (_normalize_slice(image) * 255.0).round().astype(np.uint8)
    return np.repeat(gray[..., None], repeats=3, axis=-1)


def _draw_boundary(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], width: int = 2) -> None:
    edge = _boundary(mask)
    if width > 1 and edge.any():
        structure = generate_binary_structure(edge.ndim, 1)
        for _ in range(width - 1):
            edge = binary_dilation(edge, structure=structure)
    rgb[edge] = np.asarray(color, dtype=np.uint8)


def _blend_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.65) -> None:
    if not mask.any():
        return
    color_arr = np.asarray(color, dtype=np.float32)
    rgb_float = rgb.astype(np.float32)
    rgb_float[mask] = (1.0 - alpha) * rgb_float[mask] + alpha * color_arr
    rgb[mask] = np.clip(rgb_float[mask], 0, 255).astype(np.uint8)


def _panel(base: np.ndarray, title: str, scale: int = 3) -> Image.Image:
    title_height = 28
    image = Image.fromarray(base, mode="RGB")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image.width, image.height + title_height), color=(18, 18, 18))
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(245, 245, 245))
    return canvas


def _regions_panel(base: np.ndarray, regions: dict[str, np.ndarray], slice_index: int, title: str) -> Image.Image:
    rgb = base.copy()
    colors = {"WT": "#38b000", "TC": "#ffd60a", "ET": "#ff2d55"}
    for name in ("WT", "TC", "ET"):
        _draw_boundary(rgb, regions[name][slice_index], _hex_to_rgb(colors[name]), width=2)
    return _panel(rgb, title)


def _plot_case(
    *,
    image: Tensor,
    target_regions: dict[str, np.ndarray],
    raw_regions: dict[str, np.ndarray],
    structural_regions: dict[str, np.ndarray],
    case_id: str,
    slice_index: int,
    output_path: Path,
    image_channel: int,
) -> None:
    image_np = image.detach().cpu().numpy()
    base = _gray_rgb(image_np[image_channel, slice_index])
    removed_et_fp = raw_regions["ET"] & ~structural_regions["ET"] & ~target_regions["ET"]
    preserved_tc = structural_regions["TC"] & raw_regions["TC"]
    preserved_wt = structural_regions["WT"] & raw_regions["WT"]

    panels = [
        _regions_panel(base, target_regions, slice_index, "Target regions"),
        _regions_panel(base, raw_regions, slice_index, "Before structural prior"),
        _regions_panel(base, structural_regions, slice_index, "After structural prior"),
    ]
    retained = base.copy()
    _blend_mask(retained, removed_et_fp[slice_index], _hex_to_rgb("#ff2d55"), alpha=0.70)
    _draw_boundary(retained, preserved_tc[slice_index], _hex_to_rgb("#ffd60a"), width=1)
    _draw_boundary(retained, preserved_wt[slice_index], _hex_to_rgb("#38b000"), width=1)
    panels.append(_panel(retained, "Removed ET FP; TC/WT kept"))

    pad = 8
    header_height = 26
    width = sum(panel.width for panel in panels) + pad * (len(panels) + 1)
    height = max(panel.height for panel in panels) + pad * 2 + header_height
    contact = Image.new("RGB", (width, height), color=(12, 12, 12))
    draw = ImageDraw.Draw(contact)
    draw.text((pad, 7), f"{case_id} | slice {slice_index}", fill=(245, 245, 245))
    x_offset = pad
    for panel in panels:
        contact.paste(panel, (x_offset, header_height + pad))
        x_offset += panel.width + pad
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact.save(output_path)


def _select_slice(raw_regions: dict[str, np.ndarray], structural_regions: dict[str, np.ndarray], target_regions: dict[str, np.ndarray]) -> int:
    removed_et_fp = raw_regions["ET"] & ~structural_regions["ET"] & ~target_regions["ET"]
    score = removed_et_fp.reshape(removed_et_fp.shape[0], -1).sum(axis=1)
    if int(score.max()) > 0:
        return int(score.argmax())
    target_score = target_regions["ET"].reshape(target_regions["ET"].shape[0], -1).sum(axis=1)
    if int(target_score.max()) > 0:
        return int(target_score.argmax())
    raw_score = raw_regions["WT"].reshape(raw_regions["WT"].shape[0], -1).sum(axis=1)
    return int(raw_score.argmax())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Make qualitative structural-prior figures for causal MedNeXt BraTS runs.")
    parser.add_argument("--checkpoint", default="runs/_ood_causal_adapt_brats_v5_et_precision_e2/best.pt")
    parser.add_argument("--brats-csv", default="data/brats/brats_val.csv")
    parser.add_argument("--context-csv", default="data/brats/brats_train.csv")
    parser.add_argument("--data-root", default="data/brats/archive/BraTS2020_training_data/content/data")
    parser.add_argument("--output-dir", default="runs/figures/structural_prior_qual")
    parser.add_argument("--max-volumes", type=int, default=4)
    parser.add_argument("--max-context-volumes", type=int)
    parser.add_argument("--max-context-bank-batches", type=int, default=2)
    parser.add_argument("--context-bank-size", type=int, default=4)
    parser.add_argument("--context-bank-sampling", default="farthest")
    parser.add_argument("--adjustment-contexts", type=int, default=2)
    parser.add_argument("--adjustment-context-selection", default="diverse-nearest")
    parser.add_argument("--region-thresholds", default="WT=0.65,TC=0.3,ET=0.55")
    parser.add_argument("--structural-min-component-size", type=int, default=32)
    parser.add_argument("--structural-fill-holes", action="store_true")
    parser.add_argument("--structural-keep-largest", action="store_true")
    parser.add_argument("--volume-size", type=int, default=64)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--slice-col", default="slice")
    parser.add_argument("--h5-image-key", default="image")
    parser.add_argument("--h5-mask-key", default="mask")
    parser.add_argument("--image-channel", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    thresholds = parse_region_thresholds(args.region_thresholds)
    if thresholds is None:
        raise ValueError("--region-thresholds is required for qualitative structural figures.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = _resolve_device(args.device)
    model = _build_model(checkpoint, _model_args(args)).to(device)
    model.eval()

    eval_dataset = _make_dataset(args, csv_path=args.brats_csv, max_volumes=args.max_volumes)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    context_dataset = _make_dataset(args, csv_path=args.context_csv, max_volumes=args.max_context_volumes)
    context_loader = DataLoader(context_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    context_bank = build_context_bank(
        model,
        context_loader,
        device,
        max_contexts=args.context_bank_size,
        max_batches=args.max_context_bank_batches,
        sampling=args.context_bank_sampling,
        seed=args.seed,
    ).to(device)

    output_dir = Path(args.output_dir)
    records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(eval_loader, start=1):
        image = batch["image"].to(device)
        target = batch["mask"]
        outputs = model(
            image,
            context_bank=context_bank,
            max_adjustment_contexts=args.adjustment_contexts,
            adjustment_context_selection=args.adjustment_context_selection,
        )
        logits = outputs.get("adjusted_logits", outputs["logits"])
        probs = torch.sigmoid(logits.detach().cpu())[0].numpy()
        target_sub = target[0].detach().cpu().numpy()
        raw_regions = _enforce_hierarchy(_region_masks_from_probs(probs, thresholds))
        target_regions = _target_region_masks(target_sub)
        structural_regions = _structural_masks(
            raw_regions,
            min_component_size=args.structural_min_component_size,
            fill_holes=args.structural_fill_holes,
            keep_largest=args.structural_keep_largest,
        )
        slice_index = _select_slice(raw_regions, structural_regions, target_regions)
        case_id = str(batch["case_id"][0])
        figure_path = output_dir / f"{case_id}_slice{slice_index:03d}_structural_prior.png"
        _plot_case(
            image=batch["image"][0],
            target_regions=target_regions,
            raw_regions=raw_regions,
            structural_regions=structural_regions,
            case_id=case_id,
            slice_index=slice_index,
            output_path=figure_path,
            image_channel=args.image_channel,
        )

        removed_et_fp = raw_regions["ET"] & ~structural_regions["ET"] & ~target_regions["ET"]
        raw_et_fp = raw_regions["ET"] & ~target_regions["ET"]
        raw_metrics = {name: binary_volume_metrics_from_masks(raw_regions[name], target_regions[name]) for name in ("WT", "TC", "ET")}
        structural_metrics = {
            name: binary_volume_metrics_from_masks(structural_regions[name], target_regions[name]) for name in ("WT", "TC", "ET")
        }
        records.append(
            {
                "case_id": case_id,
                "volume": int(batch["volume"][0]),
                "slice": slice_index,
                "figure": str(figure_path),
                "raw_et_false_positive_voxels": int(raw_et_fp.sum()),
                "removed_et_false_positive_voxels": int(removed_et_fp.sum()),
                "raw_wt_voxels": int(raw_regions["WT"].sum()),
                "structural_wt_voxels": int(structural_regions["WT"].sum()),
                "raw_tc_voxels": int(raw_regions["TC"].sum()),
                "structural_tc_voxels": int(structural_regions["TC"].sum()),
                "raw_et_dice": raw_metrics["ET"]["dice"],
                "structural_et_dice": structural_metrics["ET"]["dice"],
                "raw_tc_dice": raw_metrics["TC"]["dice"],
                "structural_tc_dice": structural_metrics["TC"]["dice"],
                "raw_wt_dice": raw_metrics["WT"]["dice"],
                "structural_wt_dice": structural_metrics["WT"]["dice"],
            }
        )
        if batch_index >= int(args.max_volumes):
            break

    summary = {
        "checkpoint": str(args.checkpoint),
        "region_thresholds": thresholds,
        "structural_min_component_size": int(args.structural_min_component_size),
        "records": records,
        "total_raw_et_false_positive_voxels": int(sum(item["raw_et_false_positive_voxels"] for item in records)),
        "total_removed_et_false_positive_voxels": int(sum(item["removed_et_false_positive_voxels"] for item in records)),
        "mean_raw_et_dice": float(np.mean([item["raw_et_dice"] for item in records])) if records else float("nan"),
        "mean_structural_et_dice": float(np.mean([item["structural_et_dice"] for item in records])) if records else float("nan"),
        "mean_raw_tc_dice": float(np.mean([item["raw_tc_dice"] for item in records])) if records else float("nan"),
        "mean_structural_tc_dice": float(np.mean([item["structural_tc_dice"] for item in records])) if records else float("nan"),
        "mean_raw_wt_dice": float(np.mean([item["raw_wt_dice"] for item in records])) if records else float("nan"),
        "mean_structural_wt_dice": float(np.mean([item["structural_wt_dice"] for item in records])) if records else float("nan"),
    }
    summary_path = output_dir / "summary.json"
    _save_json(summary, summary_path)
    print({"summary_json": str(summary_path), "figures": [item["figure"] for item in records]})


if __name__ == "__main__":
    main()
