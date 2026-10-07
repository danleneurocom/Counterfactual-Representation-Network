"""Harmonised preprocessing for UTSW-Glioma and BraTS 2020.

Every case goes through the same steps, in this order:

1. load the four modalities in canonical order (FLAIR, T1, T1CE, T2) and the label
   map; NIfTI volumes are reoriented to the closest canonical (RAS) orientation;
2. resample to isotropic ``--resample-mm`` (default 1 mm) when the header spacing differs;
3. compute the brain mask on the RAW intensities (any modality non-zero) -- this is the
   bug fix: the old loaders z-scored first, which made background non-zero and turned
   the foreground crop into a no-op;
4. crop the brain bounding box (+margin), zero-pad it to a cube (keeps aspect ratio and
   makes the scale identical across datasets), resize the cube to ``--size``^3;
5. z-score every modality inside the brain mask, clip to [-5, 5], background stays 0.

Outputs one ``<case_id>.npz`` per case (image float16 [4,S,S,S], label uint8 [S,S,S] with
0=background, 1=NCR/NET, 2=edema, 3=ET, brain uint8) and an ``index.csv`` with label
statistics, effective voxel spacing (for HD95 in mm), metadata proxies and per-modality
mean intensity inside ET / edema (use it to sanity-check the modality order).

Examples
--------
UTSW (registered SRI24 images, manual labels preferred)::

    python -m trace_seg3d.preprocess utsw \
        --root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
        --metadata data/brats/UTSW_Glioma_Metadata-2-1.tsv \
        --out data/processed/utsw_128

BraTS 2020 official NIfTI release (recommended)::

    python -m trace_seg3d.preprocess brats-nifti \
        --root data/brats/MICCAI_BraTS2020_TrainingData \
        --out data/processed/brats_128

BraTS 2020 Kaggle HDF5 slices (fallback; you MUST state the channel order)::

    python -m trace_seg3d.preprocess brats-h5 \
        --root data/brats/archive/BraTS2020_training_data/content/data \
        --h5-modality-order t1,t1ce,t2,flair --out data/processed/brats_128
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

CANONICAL_MODALITIES = ("flair", "t1", "t1ce", "t2")
LABEL_NAMES = {1: "ncr", 2: "ed", 3: "et"}
INDEX_COLUMNS = (
    "case_id",
    "dataset",
    "label_source",
    "size",
    "spacing_mm",
    "orig_shape",
    "orig_spacing",
    "ncr_voxels",
    "ed_voxels",
    "et_voxels",
    "wt_voxels",
    "et_present",
    "site",
    "scanner",
    "field_strength",
    "grade",
    "idh",
    "mean_in_et",
    "mean_in_ed",
)


@dataclass
class RawCase:
    case_id: str
    dataset: str
    image: np.ndarray  # [4, X, Y, Z] float32, canonical modality order
    label: np.ndarray  # [X, Y, Z] int, values {0,1,2,3}
    spacing: tuple[float, float, float]
    label_source: str = "manual"
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape {data.shape}")
    return data, zooms  # type: ignore[return-value]


def brats_labels_to_canonical(label: np.ndarray) -> np.ndarray:
    """Map BraTS label codes {1: NCR/NET, 2: ED, 4 (or 3): ET} to {1, 2, 3}."""

    label = np.asarray(label).astype(np.int16)
    out = np.zeros(label.shape, dtype=np.uint8)
    out[label == 1] = 1
    out[label == 2] = 2
    out[(label == 4) | (label == 3)] = 3
    unknown = set(np.unique(label).tolist()) - {0, 1, 2, 3, 4}
    if unknown:
        raise ValueError(f"Unexpected label values {sorted(unknown)}")
    return out


def resample_isotropic(
    image: np.ndarray, label: np.ndarray, spacing: tuple[float, float, float], target_mm: float
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    if target_mm <= 0 or all(abs(s - target_mm) / target_mm < 0.01 for s in spacing):
        return image, label, spacing
    from scipy.ndimage import zoom

    factors = tuple(s / target_mm for s in spacing)
    image = np.stack([zoom(channel, factors, order=1) for channel in image]).astype(np.float32)
    label = zoom(label, factors, order=0).astype(np.uint8)
    return image, label, (target_mm, target_mm, target_mm)


def brain_mask_from_raw(image: np.ndarray, rel_eps: float = 1e-6) -> np.ndarray:
    scale = float(np.abs(image).max()) or 1.0
    return (np.abs(image) > rel_eps * scale).any(axis=0)


def crop_pad_cube(
    image: np.ndarray, label: np.ndarray, brain: np.ndarray, margin: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not brain.any():
        raise ValueError("empty brain mask")
    coords = np.argwhere(brain)
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(brain.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    image = image[(slice(None), *sl)]
    label = label[sl]
    brain = brain[sl]
    side = int(max(image.shape[1:]))
    pads = []
    for extent in image.shape[1:]:
        before = (side - extent) // 2
        pads.append((before, side - extent - before))
    image = np.pad(image, [(0, 0), *pads])
    label = np.pad(label, pads)
    brain = np.pad(brain, pads)
    info = {"crop_lo": lo.tolist(), "crop_hi": hi.tolist(), "pad": pads, "cube_side": side}
    return image, label, brain, info


def resize_cube(image: np.ndarray, label: np.ndarray, brain: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    shape = (size, size, size)
    img_t = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
    img_t = F.interpolate(img_t, size=shape, mode="trilinear", align_corners=False)[0]
    # one-hot + trilinear + argmax is smoother than nearest for downsampling labels
    lab = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))
    onehot = F.one_hot(lab, num_classes=4).permute(3, 0, 1, 2).float().unsqueeze(0)
    onehot = F.interpolate(onehot, size=shape, mode="trilinear", align_corners=False)[0]
    lab_r = onehot.argmax(dim=0).to(torch.uint8)
    br = torch.from_numpy(np.ascontiguousarray(brain).astype(np.float32))[None, None]
    br = F.interpolate(br, size=shape, mode="trilinear", align_corners=False)[0, 0] > 0.5
    return img_t.numpy(), lab_r.numpy(), br.numpy()


def zscore_in_brain(image: np.ndarray, brain: np.ndarray, clip: float = 5.0) -> np.ndarray:
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        values = image[c][brain]
        if values.size == 0:
            continue
        mean, std = float(values.mean()), float(values.std())
        if std < 1e-6:
            continue
        channel = (image[c] - mean) / std
        channel = np.clip(channel, -clip, clip)
        channel[~brain] = 0.0
        out[c] = channel
    return out


def harmonise(case: RawCase, size: int, resample_mm: float, margin: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    image, label, spacing = resample_isotropic(case.image, case.label, case.spacing, resample_mm)
    brain = brain_mask_from_raw(image)
    orig_shape = list(case.image.shape[1:])
    image, label, brain, crop_info = crop_pad_cube(image, label, brain, margin)
    side_mm = crop_info["cube_side"] * float(np.mean(spacing))
    image, label, brain = resize_cube(image, label, brain, size)
    image = zscore_in_brain(image, brain)
    eff_spacing = side_mm / float(size)

    stats: dict[str, Any] = {
        "case_id": case.case_id,
        "dataset": case.dataset,
        "label_source": case.label_source,
        "size": size,
        "spacing_mm": round(eff_spacing, 4),
        "orig_shape": "x".join(str(v) for v in orig_shape),
        "orig_spacing": "x".join(f"{v:.3f}" for v in case.spacing),
        "ncr_voxels": int((label == 1).sum()),
        "ed_voxels": int((label == 2).sum()),
        "et_voxels": int((label == 3).sum()),
        "wt_voxels": int((label > 0).sum()),
    }
    stats["et_present"] = int(stats["et_voxels"] > 0)
    for key in ("site", "scanner", "field_strength", "grade", "idh"):
        stats[key] = case.meta.get(key, "NA")
    for name, code in (("mean_in_et", 3), ("mean_in_ed", 2)):
        region = label == code
        if region.any():
            stats[name] = "|".join(f"{m}:{float(image[i][region].mean()):.2f}" for i, m in enumerate(CANONICAL_MODALITIES))
        else:
            stats[name] = "NA"
    meta = {**case.meta, **crop_info, **stats, "modalities": list(CANONICAL_MODALITIES)}
    arrays = {
        "image": image.astype(np.float16),
        "label": label.astype(np.uint8),
        "brain": brain.astype(np.uint8),
    }
    return arrays, meta


def save_case(out_dir: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{meta['case_id']}.npz", meta=json.dumps(meta, default=str), **arrays)


def load_case(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "meta"}
        meta = json.loads(str(data["meta"]))
    return arrays, meta


# ----------------------------------------------------------------------------- UTSW


def _clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "NA" if text.lower() in {"", "nan", "none", "na", "n/a"} else text


def _utsw_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not Path(path).exists():
        return {}
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    rows: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        sid = _clean(row.get("Subject ID"))
        make = _clean(row.get("Scanner Make"))
        model = _clean(row.get("Scanner Model"))
        strength = _clean(row.get("Scanner Strength"))
        rows[sid] = {
            "site": "UTSW",
            "scanner": f"{make}/{model}",
            "scanner_make": make,
            "field_strength": strength,
            "grade": _clean(row.get("Tumor Grade")),
            "idh": _clean(row.get("IDH")),
            "tumor_type": _clean(row.get("Tumor Type")),
            "operation_status": _clean(row.get("Operation Status")),
        }
    return rows


def _first_existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def utsw_case_dirs(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).iterdir() if p.is_dir())


def load_utsw_case(case_dir: Path, use_ants: bool, metadata: dict[str, dict[str, str]]) -> RawCase:
    vols = []
    spacing = None
    ref_shape = None
    for modality in CANONICAL_MODALITIES:
        ants_name = "fl" if modality == "flair" else modality
        candidates = [case_dir / f"brain_{ants_name}_ants.nii.gz"] if use_ants else []
        candidates += [case_dir / f"brain_{modality}.nii.gz", case_dir / f"brain_{modality.upper()}.nii.gz"]
        found = _first_existing(candidates)
        if not found:
            raise FileNotFoundError(f"{case_dir.name}: missing {modality}")
        data, zooms = _load_nifti(found[0])
        if ref_shape is None:
            ref_shape, spacing = data.shape, zooms
        elif data.shape != ref_shape:
            raise ValueError(f"{case_dir.name}: modality shapes differ ({data.shape} vs {ref_shape})")
        vols.append(data)
    seg_candidates = [
        (case_dir / "rtumorseg_manual_correction.nii.gz", "manual"),
        (case_dir / "tumorseg_manual_correction.nii.gz", "manual"),
        (case_dir / "tumorseg_FeTS.nii.gz", "auto_fets"),
    ]
    label = None
    label_source = "missing"
    for path, source in seg_candidates:
        if not path.exists():
            continue
        seg, _ = _load_nifti(path)
        if seg.shape == ref_shape:
            label, label_source = brats_labels_to_canonical(np.rint(seg)), source
            break
    if label is None:
        raise FileNotFoundError(f"{case_dir.name}: no segmentation matching image shape {ref_shape}")
    meta = dict(metadata.get(case_dir.name, {"site": "UTSW"}))
    meta["space"] = "ants" if use_ants else "native"
    return RawCase(case_dir.name, "utsw", np.stack(vols), label, spacing, label_source, meta)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- BraTS NIfTI


def _brats_name_mapping(root: Path) -> dict[str, dict[str, str]]:
    path = root / "name_mapping.csv"
    if not path.exists():
        return {}
    import pandas as pd

    frame = pd.read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        case = _clean(row.get("BraTS_2020_subject_ID"))
        b19 = _clean(row.get("BraTS_2019_subject_ID"))
        tcga = _clean(row.get("TCGA_TCIA_subject_ID"))
        # BraTS19_<INSTITUTION>_<id>_1 -> institution token (CBICA, TCIA01..13, 2013, ...)
        site = "NA"
        match = re.match(r"BraTS19_([^_]+)_", b19)
        if match:
            site = match.group(1)
        elif tcga != "NA":
            site = "TCGA"
        out[case] = {
            "site": site,
            "scanner": "NA",
            "field_strength": "NA",
            "grade": _clean(row.get("Grade")),
            "idh": "NA",
            "tcga_id": tcga,
        }
    return out


def brats_case_dirs(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name.startswith("BraTS"))


def load_brats_nifti_case(case_dir: Path, mapping: dict[str, dict[str, str]]) -> RawCase:
    vols = []
    spacing = None
    for modality in CANONICAL_MODALITIES:
        found = sorted(case_dir.glob(f"*_{modality}.nii*"))
        if not found:
            raise FileNotFoundError(f"{case_dir.name}: missing {modality}")
        data, zooms = _load_nifti(found[0])
        spacing = spacing or zooms
        vols.append(data)
    # BraTS20_Training_355 ships its label as 'W39_1998.09.19_Segm.nii'
    seg_files = sorted(case_dir.glob("*_seg.nii*")) or sorted(case_dir.glob("*[Ss]eg*.nii*"))
    if not seg_files:
        raise FileNotFoundError(f"{case_dir.name}: missing seg")
    seg, _ = _load_nifti(seg_files[0])
    label = brats_labels_to_canonical(np.rint(seg))
    meta = dict(mapping.get(case_dir.name, {"site": "NA", "grade": "NA"}))
    return RawCase(case_dir.name, "brats", np.stack(vols), label, spacing, "manual", meta)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- BraTS Kaggle HDF5


def brats_h5_volumes(root: Path) -> dict[int, list[tuple[int, Path]]]:
    volumes: dict[int, list[tuple[int, Path]]] = {}
    for path in Path(root).glob("volume_*_slice_*.h5"):
        match = re.match(r"volume_(\d+)_slice_(\d+)\.h5", path.name)
        if match:
            volumes.setdefault(int(match.group(1)), []).append((int(match.group(2)), path))
    return {vid: sorted(items) for vid, items in sorted(volumes.items())}


def load_brats_h5_case(
    volume_id: int,
    slices: list[tuple[int, Path]],
    modality_order: tuple[str, ...],
    mask_order: tuple[str, ...],
    flip_axes: tuple[int, ...],
) -> RawCase:
    import h5py

    images, masks = [], []
    for _, path in slices:
        with h5py.File(path, "r") as handle:
            img = np.asarray(handle["image"], dtype=np.float32)
            msk = np.asarray(handle["mask"], dtype=np.float32)
        if img.shape[-1] == 4:
            img = np.moveaxis(img, -1, 0)
        if msk.ndim == 3 and msk.shape[-1] == 3:
            msk = np.moveaxis(msk, -1, 0)
        images.append(img)
        masks.append(msk)
    image = np.stack(images, axis=-1)  # [4, H, W, Z]
    mask = np.stack(masks, axis=-1)
    reorder = [modality_order.index(m) for m in CANONICAL_MODALITIES]
    image = image[reorder]
    label = np.zeros(image.shape[1:], dtype=np.uint8)
    if mask.ndim == 4:
        code = {"ncr": 1, "ed": 2, "et": 3}
        for channel, name in enumerate(mask_order):
            label[mask[channel] > 0.5] = code[name]
    else:
        label = brats_labels_to_canonical(np.rint(mask))
    for axis in flip_axes:
        image = np.flip(image, axis=axis + 1)
        label = np.flip(label, axis=axis)
    # the Kaggle export may already be intensity-normalised with a non-zero background:
    # re-zero voxels equal to the corner (background) value so the raw brain mask works
    background = np.median(image[:, 0, 0, :], axis=1)
    for c in range(4):
        if background[c] != 0.0:
            image[c][np.isclose(image[c], background[c])] = 0.0
    return RawCase(f"volume_{volume_id:03d}", "brats", np.ascontiguousarray(image), np.ascontiguousarray(label), (1.0, 1.0, 1.0), "manual", {"site": "NA", "grade": "NA"})


# ----------------------------------------------------------------------------- driver


def _process_one(job: tuple[Callable[[], RawCase], Path, int, float, int, bool]) -> dict[str, Any] | None:
    loader, out_dir, size, resample_mm, margin, overwrite = job
    try:
        case = loader()
        target = out_dir / f"{case.case_id}.npz"
        if target.exists() and not overwrite:
            _, meta = load_case(target)
            return {k: meta.get(k, "NA") for k in INDEX_COLUMNS}
        arrays, meta = harmonise(case, size, resample_mm, margin)
        save_case(out_dir, arrays, meta)
        return {k: meta.get(k, "NA") for k in INDEX_COLUMNS}
    except Exception as exc:  # keep going, report at the end
        return {"case_id": f"ERROR::{exc}"}


class _Loader:
    """Picklable deferred loader for multiprocessing."""

    def __init__(self, fn: Callable[..., RawCase], *args: Any) -> None:
        self.fn, self.args = fn, args

    def __call__(self) -> RawCase:
        return self.fn(*self.args)


def run(loaders: list[_Loader], out_dir: Path, size: int, resample_mm: float, margin: int, workers: int, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(loader, out_dir, size, resample_mm, margin, overwrite) for loader in loaders]
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_process_one, jobs, chunksize=1)
    else:
        rows = [_process_one(job) for job in jobs]
    errors = [r["case_id"] for r in rows if r and str(r["case_id"]).startswith("ERROR::")]
    good = [r for r in rows if r and not str(r["case_id"]).startswith("ERROR::")]
    with open(out_dir / "index.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(sorted(good, key=lambda r: str(r["case_id"])))
    print(f"processed {len(good)} cases -> {out_dir}")
    if good:
        et_empty = sum(1 for r in good if int(r["et_present"]) == 0)
        auto = sum(1 for r in good if r["label_source"] != "manual")
        print(f"  ET-empty cases: {et_empty}/{len(good)}   non-manual labels: {auto}")
    if errors:
        print(f"  {len(errors)} failures:")
        for err in errors[:20]:
            print("   ", err)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--size", type=int, default=128)
        p.add_argument("--resample-mm", type=float, default=1.0)
        p.add_argument("--margin", type=int, default=4)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--limit", type=int)
        p.add_argument("--overwrite", action="store_true")

    p_utsw = sub.add_parser("utsw")
    common(p_utsw)
    p_utsw.add_argument("--metadata", type=Path)
    p_utsw.add_argument("--native", action="store_true", help="use native-space images instead of the SRI24 ANTs registrations")

    p_bn = sub.add_parser("brats-nifti")
    common(p_bn)

    p_bh = sub.add_parser("brats-h5")
    common(p_bh)
    p_bh.add_argument("--h5-modality-order", required=True, help="channel order inside the h5 'image', e.g. t1,t1ce,t2,flair")
    p_bh.add_argument("--h5-mask-order", default="ncr,ed,et")
    p_bh.add_argument("--flip-axes", default="", help="comma list of spatial axes (0,1,2) to flip to match NIfTI/RAS orientation")

    args = parser.parse_args(argv)
    loaders: list[_Loader] = []
    if args.source == "utsw":
        metadata = _utsw_metadata(args.metadata)
        dirs = utsw_case_dirs(args.root)[: args.limit]
        loaders = [_Loader(load_utsw_case, d, not args.native, metadata) for d in dirs]
    elif args.source == "brats-nifti":
        mapping = _brats_name_mapping(args.root)
        dirs = brats_case_dirs(args.root)[: args.limit]
        loaders = [_Loader(load_brats_nifti_case, d, mapping) for d in dirs]
    else:
        order = tuple(s.strip().lower() for s in args.h5_modality_order.split(","))
        if sorted(order) != sorted(CANONICAL_MODALITIES):
            raise SystemExit(f"--h5-modality-order must be a permutation of {CANONICAL_MODALITIES}")
        mask_order = tuple(s.strip().lower() for s in args.h5_mask_order.split(","))
        flips = tuple(int(s) for s in args.flip_axes.split(",") if s.strip())
        volumes = list(brats_h5_volumes(args.root).items())[: args.limit]
        loaders = [_Loader(load_brats_h5_case, vid, sl, order, mask_order, flips) for vid, sl in volumes]
    run(loaders, args.out, args.size, args.resample_mm, args.margin, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
