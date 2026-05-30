#!/usr/bin/env python3
"""Preprocess BraTS20 .nii volumes into per-volume .h5 files for CRN training.

Each output .h5 contains:
  - image: (4, D, H, W) float32  [flair, t1, t1ce, t2]
  - mask:  (3, D, H, W) uint8    [ncr_net, edema, enhancing_tumor]

The script also writes a metadata CSV with path / mask / target / volume / slice.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


MODALITIES = ["flair", "t1", "t1ce", "t2"]


def normalize_minmax(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_v = float(np.nanmin(array))
    max_v = float(np.nanmax(array))
    if max_v <= min_v:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_v) / (max_v - min_v)


def resize_volume(array: np.ndarray, spatial_size: tuple[int, int]) -> np.ndarray:
    """Resize (C, D, H, W) or (D, H, W) spatially to spatial_size.
    Uses trilinear for images, nearest for masks.
    """
    is_mask = array.dtype == np.uint8
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)  # (1, C, D, H, W) or (1, 1, D, H, W)
    if tensor.ndim == 5 and tensor.shape[1] == 1:
        pass
    target_size = (tensor.shape[-3], *spatial_size)
    mode = "nearest" if is_mask else "trilinear"
    kwargs = {} if is_mask else {"align_corners": False}
    resized = F.interpolate(tensor, size=target_size, mode=mode, **kwargs)
    out = resized.squeeze(0).numpy()
    if is_mask:
        out = (out > 0.5).astype(np.uint8)
    return out


def process_case(case_dir: Path, spatial_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    """Return image (4, D, H, W) float32 and mask (3, D, H, W) uint8 or None."""
    case_name = case_dir.name
    images = []
    for mod in MODALITIES:
        mod_path = case_dir / f"{case_name}_{mod}.nii"
        if not mod_path.exists():
            mod_path = mod_path.with_suffix(".nii.gz")
        img = nib.load(str(mod_path)).get_fdata()
        img = normalize_minmax(img)
        images.append(img)
    image = np.stack(images, axis=0)  # (4, H, W, D)  nibabel default is (H,W,D)
    # nibabel get_fdata for .nii is usually (H, W, D)
    # We want (4, D, H, W)
    image = np.transpose(image, (0, 3, 1, 2))  # (4, D, H, W)
    image = resize_volume(image, spatial_size)

    seg_path = case_dir / f"{case_name}_seg.nii"
    if not seg_path.exists():
        seg_path = seg_path.with_suffix(".nii.gz")
    if not seg_path.exists():
        # Some BraTS cases use alternative segmentation filenames
        for candidate in case_dir.glob("*.nii*"):
            if candidate.name.startswith(case_name):
                continue
            if "seg" in candidate.name.lower() or "segm" in candidate.name.lower():
                seg_path = candidate
                break
    if seg_path.exists():
        seg = nib.load(str(seg_path)).get_fdata().astype(np.uint8)
        seg = np.transpose(seg, (2, 0, 1))  # (D, H, W)
        # BraTS labels: 0=bg, 1=NCR/NET, 2=ED, 4=ET
        mask = np.stack([
            (seg == 1).astype(np.uint8),
            (seg == 2).astype(np.uint8),
            (seg == 4).astype(np.uint8),
        ], axis=0)  # (3, D, H, W)
        mask = resize_volume(mask, spatial_size)
    else:
        mask = None

    return image, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Root with BraTS2020_TrainingData / BraTS2020_ValidationData")
    parser.add_argument("--output-dir", default="data/brats20_processed")
    parser.add_argument("--spatial-size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)
    spatial_size = tuple(args.spatial_size)

    # Training
    train_root = input_dir / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    train_cases = sorted([d for d in train_root.iterdir() if d.is_dir()])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_cases)
    val_fraction = 0.2
    val_count = max(1, round(len(train_cases) * val_fraction))
    val_cases = train_cases[:val_count]
    train_cases = train_cases[val_count:]

    rows: list[dict] = []

    def write_split(cases: list[Path], split: str) -> None:
        for vol_idx, case_dir in enumerate(tqdm(cases, desc=f"Processing {split}")):
            image, mask = process_case(case_dir, spatial_size)
            out_path = output_dir / split / f"{case_dir.name}.h5"
            with h5py.File(out_path, "w") as f:
                f.create_dataset("image", data=image, compression="gzip")
                if mask is not None:
                    f.create_dataset("mask", data=mask, compression="gzip")
            # target = 1 if any tumor present
            target = int((mask > 0).any()) if mask is not None else 0
            for s in range(image.shape[1]):
                rows.append({
                    "path": str(out_path.relative_to(output_dir)),
                    "mask": str(out_path.relative_to(output_dir)),
                    "target": target,
                    "volume": vol_idx,
                    "slice": s,
                })

    write_split(train_cases, "train")
    write_split(val_cases, "val")

    frame = pd.DataFrame(rows)
    train_frame = frame[frame["path"].str.contains("/train/")].reset_index(drop=True)
    val_frame = frame[frame["path"].str.contains("/val/")].reset_index(drop=True)
    train_frame.to_csv(output_dir / "brats_train.csv", index=False)
    val_frame.to_csv(output_dir / "brats_val.csv", index=False)
    print(f"Saved {len(train_frame)} train rows, {len(val_frame)} val rows to {output_dir}")


if __name__ == "__main__":
    main()
