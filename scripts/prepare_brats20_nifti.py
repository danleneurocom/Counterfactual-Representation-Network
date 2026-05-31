from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


MODALITIES = ("t1", "t1ce", "t2", "flair")
SEG_LABELS = {
    "ncr_net": 1,
    "edema": 2,
    "enhancing_tumor": 4,
}


def _find_modality(case_dir: Path, modality: str) -> Path | None:
    patterns = [f"*_{modality}.nii*", f"*_{modality.upper()}.nii*"]
    for pattern in patterns:
        hits = sorted(case_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _find_segmentation(case_dir: Path) -> Path | None:
    for pattern in ("*_seg.nii*", "*_SEG.nii*"):
        hits = sorted(case_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape} from {path}")
    return np.moveaxis(data, -1, 0)


def _minmax_normalize(volume: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(volume))
    hi = float(np.nanmax(volume))
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - lo) / (hi - lo)


def _build_subregion_mask(seg: np.ndarray) -> np.ndarray:
    ncr = (seg == SEG_LABELS["ncr_net"]).astype(np.float32)
    edema = (seg == SEG_LABELS["edema"]).astype(np.float32)
    et = (seg == SEG_LABELS["enhancing_tumor"]).astype(np.float32)
    return np.stack([ncr, edema, et], axis=0)


def _volume_id_from_name(case_name: str, fallback: int) -> int:
    match = re.search(r"_(\d+)$", case_name)
    if match:
        return int(match.group(1))
    return fallback


def _collect_cases(data_root: Path) -> list[Path]:
    candidates = sorted(data_root.glob("**/BraTS20_Training_*"))
    return [path for path in candidates if path.is_dir()]


def _split_by_volume(frame: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    volumes = frame["volume"].drop_duplicates().to_numpy(copy=True)
    rng.shuffle(volumes)
    val_count = max(1, round(len(volumes) * val_fraction))
    val_volumes = set(volumes[:val_count])
    val_mask = frame["volume"].isin(val_volumes)
    return frame.loc[~val_mask].reset_index(drop=True), frame.loc[val_mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BraTS20 NIfTI volumes into per-slice HDF5 and CSV metadata.")
    parser.add_argument("--data-root", required=True, help="Root directory containing BraTS2020 training cases.")
    parser.add_argument("--output-dir", default="data/brats", help="Output directory for H5 slices and CSVs.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-cases", type=int, default=0, help="Optional cap on number of cases to convert.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    slices_dir = output_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = _collect_cases(data_root)
    if args.limit_cases and args.limit_cases > 0:
        case_dirs = case_dirs[: int(args.limit_cases)]
    if not case_dirs:
        raise SystemExit(f"No BraTS20 training cases found under {data_root}")

    rows: list[dict[str, object]] = []
    for index, case_dir in enumerate(case_dirs, start=1):
        case_name = case_dir.name
        modality_paths = {m: _find_modality(case_dir, m) for m in MODALITIES}
        seg_path = _find_segmentation(case_dir)
        if any(path is None for path in modality_paths.values()) or seg_path is None:
            print(f"Skipping {case_name}: missing modality or segmentation")
            continue

        volumes = np.stack([
            _minmax_normalize(_load_nifti(modality_paths[m]))
            for m in MODALITIES
        ], axis=0)
        seg_volume = _load_nifti(seg_path)
        seg_subregions = _build_subregion_mask(seg_volume)

        depth = volumes.shape[1]
        volume_id = _volume_id_from_name(case_name, index)
        for slice_idx in range(depth):
            h5_path = slices_dir / f"{case_name}_slice_{slice_idx:03d}.h5"
            image_slice = volumes[:, slice_idx, :, :]
            mask_slice = seg_subregions[:, slice_idx, :, :]
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("image", data=image_slice.astype(np.float32), compression="gzip")
                handle.create_dataset("mask", data=mask_slice.astype(np.float32), compression="gzip")
            rows.append(
                {
                    "volume": volume_id,
                    "volume_name": case_name,
                    "slice": slice_idx,
                    "path": str(h5_path),
                    "mask": str(h5_path),
                    "target": int(mask_slice.sum() > 0),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No slices were written. Check that BraTS20 data is present and readable.")

    train, val = _split_by_volume(frame, args.val_fraction, args.seed)
    train_path = output_dir / "brats_train.csv"
    val_path = output_dir / "brats_val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    print(f"slices: {len(frame)}")
    print(f"volumes: {frame['volume'].nunique()}")
    print(f"train: {len(train)} rows -> {train_path}")
    print(f"val: {len(val)} rows -> {val_path}")


if __name__ == "__main__":
    main()
