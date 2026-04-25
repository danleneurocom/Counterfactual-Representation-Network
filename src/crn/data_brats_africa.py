"""BraTS-Africa (BraTS-SSA) zero-shot OOD loader — Phase C.3.

BraTS-Africa (Adewole et al. 2023, arXiv:2305.19369; dataset card RSNA/ASNR/MICCAI
BraTS-SSA, DOI 10.1148/ryai.240528) distributes ~95 adult-glioma MRI cases from
sub-Saharan Africa as NIfTI volumes with four modalities (T1, T1ce, T2, FLAIR)
and a segmentation mask. This module's sole responsibility is to convert those
NIfTI volumes into the per-slice HDF5 layout consumed by `ImagingCsvDataset`
(keys `image` of shape `(4, H, W)` and `mask` of shape `(3, H, W)` with
subregion channels `ncr_net / edema / enhancing_tumor`), and to write a CSV
with columns `volume, slice, path, mask, ...` identical to `brats_val.csv`.

The main pipeline is then used unchanged for inference:

    PYTHONPATH=src python -m crn.evaluate \
        --checkpoint runs/...mednextL/best.pt \
        --split val --config configs/eval_ood_brats_africa.yaml

No training is performed on BraTS-Africa: this is held-out zero-shot
evaluation used to report Dice drop Δ vs the source domain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


BRATS_AFRICA_MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")
BRATS_AFRICA_SEGMENTATION_LABELS: dict[str, int] = {
    "ncr_net": 1,   # necrotic / non-enhancing tumor core
    "edema": 2,     # peritumoral edema
    "enhancing_tumor": 4,  # GD-enhancing tumor (relabeled from 4 to 3 in newer releases)
}


def _find_modality(case_dir: Path, modality: str) -> Path | None:
    """Locate a modality NIfTI file within a case directory.

    Looks for filenames containing the modality token, case-insensitively, with
    `.nii` or `.nii.gz` suffix. BraTS-Africa files typically follow the naming
    convention `<subject>_<modality>.nii.gz` but we stay permissive.
    """
    patterns = [f"*{modality}*.nii.gz", f"*{modality.upper()}*.nii.gz", f"*{modality}*.nii"]
    for pattern in patterns:
        hits = sorted(case_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _find_segmentation(case_dir: Path) -> Path | None:
    for pattern in ("*seg*.nii.gz", "*SEG*.nii.gz", "*seg*.nii"):
        hits = sorted(case_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib  # imported lazily; nibabel is optional at runtime

    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _minmax_normalize(volume: np.ndarray) -> np.ndarray:
    lo = float(volume.min())
    hi = float(volume.max())
    if hi <= lo:
        return np.zeros_like(volume)
    return (volume - lo) / (hi - lo)


def _build_subregion_mask(seg: np.ndarray) -> np.ndarray:
    """Return a (3, D, H, W) one-hot mask with channels (ncr_net, edema, enhancing_tumor).

    BraTS-Africa uses the canonical labels {1, 2, 4} — we also accept {1, 2, 3}
    for the post-2023 relabeled release.
    """
    ncr = (seg == BRATS_AFRICA_SEGMENTATION_LABELS["ncr_net"]).astype(np.float32)
    edema = (seg == BRATS_AFRICA_SEGMENTATION_LABELS["edema"]).astype(np.float32)
    et_label = BRATS_AFRICA_SEGMENTATION_LABELS["enhancing_tumor"]
    et = ((seg == et_label) | (seg == 3)).astype(np.float32)
    return np.stack([ncr, edema, et], axis=0)


def convert_case(
    case_dir: Path,
    output_dir: Path,
    resize_to: tuple[int, int] | None = None,
) -> list[dict[str, object]]:
    """Convert a single BraTS-Africa NIfTI case into per-slice HDF5 files.

    Writes `<subject>_slice_<index>.h5` files under `output_dir` and returns a
    list of metadata rows (one per slice) ready to be concatenated into a CSV.
    """
    subject = case_dir.name
    modality_paths = {m: _find_modality(case_dir, m) for m in BRATS_AFRICA_MODALITY_ORDER}
    seg_path = _find_segmentation(case_dir)
    if any(p is None for p in modality_paths.values()) or seg_path is None:
        return []

    volumes = np.stack([
        _minmax_normalize(_load_nifti(modality_paths[m]))
        for m in BRATS_AFRICA_MODALITY_ORDER
    ], axis=0)  # (4, D, H, W)
    seg_volume = _load_nifti(seg_path)
    seg_subregions = _build_subregion_mask(seg_volume)  # (3, D, H, W)

    if resize_to is not None:
        from scipy.ndimage import zoom

        _, d, h, w = volumes.shape
        scale_h = resize_to[0] / h
        scale_w = resize_to[1] / w
        volumes = zoom(volumes, (1.0, 1.0, scale_h, scale_w), order=1)
        seg_subregions = zoom(seg_subregions, (1.0, 1.0, scale_h, scale_w), order=0)

    depth = volumes.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for slice_idx in range(depth):
        h5_path = output_dir / f"{subject}_slice_{slice_idx:03d}.h5"
        image_slice = volumes[:, slice_idx, :, :]           # (4, H, W)
        mask_slice = seg_subregions[:, slice_idx, :, :]     # (3, H, W)
        with h5py.File(h5_path, "w") as fh:
            fh.create_dataset("image", data=image_slice.astype(np.float32), compression="gzip")
            fh.create_dataset("mask", data=mask_slice.astype(np.float32), compression="gzip")
        rows.append(
            {
                "volume": subject,
                "slice": slice_idx,
                "path": str(h5_path),
                "mask": str(h5_path),
                "label0_pxl_cnt": int(mask_slice[0].sum()),
                "label1_pxl_cnt": int(mask_slice[1].sum()),
                "label2_pxl_cnt": int(mask_slice[2].sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BraTS-Africa NIfTI volumes into the CRN per-slice HDF5 layout.")
    parser.add_argument("--data-root", required=True, help="Directory containing one subdirectory per case.")
    parser.add_argument("--output-dir", default="data/brats_africa", help="Where to write H5 slices and the test CSV.")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), help="Optional H W to resize each slice to.")
    parser.add_argument("--csv-name", default="brats_africa_test.csv")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for case_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        all_rows.extend(
            convert_case(case_dir, output_dir / "slices", resize_to=tuple(args.resize) if args.resize else None)
        )
    if not all_rows:
        raise SystemExit(f"No BraTS-Africa cases found under {data_root!s}.")
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / args.csv_name
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} slices across {df['volume'].nunique()} cases to {csv_path}.")


if __name__ == "__main__":
    main()
