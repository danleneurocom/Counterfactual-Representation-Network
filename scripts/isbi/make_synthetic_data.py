"""Create tiny synthetic UTSW-like and BraTS-like raw datasets for smoke tests (NOT real data).

Two "sites" with different acquisition style (gamma, bias field, contrast, noise) and the
same tumour generator (edema shell, necrotic core, enhancing rim; ~25% of cases without ET).

    python scripts/isbi/make_synthetic_data.py --out /tmp/synth --n-utsw 24 --n-brats 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def _ellipsoid(shape, center, radii):
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    d = sum(((g - c) / r) ** 2 for g, c, r in zip(grids, center, radii))
    return d <= 1.0


def make_case(rng: np.random.Generator, shape=(72, 80, 64), style: str = "utsw"):
    brain = _ellipsoid(shape, [s / 2 for s in shape], [s * 0.42 for s in shape])
    label = np.zeros(shape, np.uint8)
    center = [s / 2 + rng.uniform(-0.12, 0.12) * s for s in shape]
    r = rng.uniform(7, 13)
    edema = _ellipsoid(shape, center, [r * rng.uniform(0.8, 1.2) for _ in shape]) & brain
    core = _ellipsoid(shape, center, [r * 0.55 for _ in shape]) & brain
    necrosis = _ellipsoid(shape, center, [r * 0.3 for _ in shape]) & brain
    has_et = rng.random() > 0.25
    label[edema] = 2
    if has_et:
        label[core] = 4
        label[necrosis] = 1
    else:
        label[core] = 1
    tissue = rng.normal(0.0, 0.05, shape)
    # canonical modality signatures (flair, t1, t1ce, t2): ET bright on t1ce, edema bright on flair/t2
    base = {"flair": 0.5, "t1": 0.6, "t1ce": 0.55, "t2": 0.45}
    gain = {
        "flair": {1: 0.2, 2: 0.45, 4: 0.3},
        "t1": {1: -0.25, 2: -0.1, 4: -0.05},
        "t1ce": {1: -0.2, 2: 0.0, 4: 0.55},
        "t2": {1: 0.5, 2: 0.4, 4: 0.25},
    }
    vols = {}
    for m in ("flair", "t1", "t1ce", "t2"):
        v = np.full(shape, base[m]) + tissue
        for code, g in gain[m].items():
            v[label == code] += g
        if style == "brats":
            v = np.clip(v, 1e-3, None) ** 1.6
            zz = np.linspace(0.75, 1.25, shape[2])[None, None, :]
            v = v * zz * 900 + rng.normal(0, 25, shape)
        else:
            v = v * 400 + rng.normal(0, 8, shape)
        v[~brain] = 0.0
        vols[m] = v.astype(np.float32)
    return vols, label, has_et


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-utsw", type=int, default=24)
    p.add_argument("--n-brats", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    affine = np.eye(4)
    utsw_root = args.out / "PKG - UTSW-Glioma" / "UTSW-Glioma"
    rows = ["Subject ID\tScanner Make\tScanner Model\tScanner Strength\tTumor Grade\tIDH\tTumor Type\tOperation Status"]
    for i in range(args.n_utsw):
        vols, label, has_et = make_case(rng, style="utsw")
        case = utsw_root / f"BT{i + 1:04d}"
        case.mkdir(parents=True, exist_ok=True)
        for m, v in vols.items():
            name = "fl" if m == "flair" else m
            nib.save(nib.Nifti1Image(v, affine), case / f"brain_{name}_ants.nii.gz")
        nib.save(nib.Nifti1Image(label.astype(np.int16), affine), case / "rtumorseg_manual_correction.nii.gz")
        make = "GE" if i % 2 else "Siemens"
        rows.append(f"BT{i + 1:04d}\t{make}\tModel{i % 3}\t{3.0 if i % 3 else 1.5}\t{4 if has_et else 2}\t{'wt' if has_et else 'mut'}\tGlioma\tPreop")
    (args.out / "UTSW_Glioma_Metadata-2-1.tsv").write_text("\n".join(rows) + "\n")
    brats_root = args.out / "MICCAI_BraTS2020_TrainingData"
    mapping = ["Grade,BraTS_2017_subject_ID,BraTS_2018_subject_ID,TCGA_TCIA_subject_ID,BraTS_2019_subject_ID,BraTS_2020_subject_ID"]
    sites = ["CBICA", "TCIA01", "2013"]
    for i in range(args.n_brats):
        vols, label, has_et = make_case(rng, style="brats")
        cid = f"BraTS20_Training_{i + 1:03d}"
        case = brats_root / cid
        case.mkdir(parents=True, exist_ok=True)
        aff = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS-like header -> exercises reorientation
        for m, v in vols.items():
            nib.save(nib.Nifti1Image(v[::-1, ::-1, :].copy(), aff), case / f"{cid}_{m}.nii.gz")
        nib.save(nib.Nifti1Image(label[::-1, ::-1, :].astype(np.int16).copy(), aff), case / f"{cid}_seg.nii.gz")
        mapping.append(f"{'HGG' if has_et else 'LGG'},NA,NA,NA,BraTS19_{sites[i % 3]}_{i:03d}_1,{cid}")
    (brats_root / "name_mapping.csv").write_text("\n".join(mapping) + "\n")
    print(f"synthetic data -> {args.out}")


if __name__ == "__main__":
    main()
