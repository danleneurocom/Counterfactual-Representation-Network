"""Sanity checks after preprocessing -- run this BEFORE training and look at the PNG.

* modality order: inside ET the T1CE channel must be the brightest (z-score); inside edema
  FLAIR/T2 are bright and T1 is dark. A wrong order in the BraTS h5 export shows up here.
* orientation: mid-slices of a few cases from each dataset side by side (same view should
  look the same way up / left-right).
* label statistics: ET-empty cases, non-manual (FeTS) labels, effective spacing.

    python -m trace_seg3d.check_data data/processed/utsw_128 data/processed/brats_128 --png check.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from trace_seg3d.preprocess import CANONICAL_MODALITIES, load_case


def _parse_means(text: str) -> dict[str, float] | None:
    if not text or text == "NA":
        return None
    return {k: float(v) for k, v in (item.split(":") for item in text.split("|"))}


def report(data_dir: Path) -> list[str]:
    with open(data_dir / "index.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lines = [f"== {data_dir} : {len(rows)} cases"]
    if not rows:
        return lines
    et_empty = sum(1 for r in rows if r["et_present"] == "0")
    sources: dict[str, int] = {}
    for r in rows:
        sources[r["label_source"]] = sources.get(r["label_source"], 0) + 1
    spacing = np.array([float(r["spacing_mm"]) for r in rows])
    lines.append(f"   ET-empty: {et_empty}  label sources: {sources}  effective spacing mm: {spacing.min():.2f}-{spacing.max():.2f}")
    for region, expect in (("mean_in_et", "t1ce"), ("mean_in_ed", None)):
        parsed = [m for m in (_parse_means(r[region]) for r in rows) if m]
        if not parsed:
            continue
        med = {k: float(np.median([m[k] for m in parsed])) for k in CANONICAL_MODALITIES}
        txt = "  ".join(f"{k}={v:+.2f}" for k, v in med.items())
        lines.append(f"   median z in {region[8:].upper()}: {txt}")
        if expect:
            frac = np.mean([max(m, key=m.get) == expect for m in parsed])
            flag = "OK" if frac > 0.7 else "!! CHECK MODALITY ORDER"
            lines.append(f"   T1CE brightest inside ET in {frac:.0%} of cases  {flag}")
    return lines


def montage(dirs: list[Path], out: Path, n: int = 3) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(dirs) * n, 4, figsize=(10, 2.6 * len(dirs) * n))
    axes = np.atleast_2d(axes)
    row = 0
    for d in dirs:
        files = sorted(d.glob("*.npz"))[:n]
        for f in files:
            arrays, meta = load_case(f)
            img, lab = arrays["image"].astype(np.float32), arrays["label"]
            c = img.shape[-1] // 2
            zc = int(np.argmax((lab > 0).sum(axis=(0, 1)))) if lab.any() else c
            views = [img[0][:, :, zc], img[2][:, :, zc], lab[:, :, zc], img[0][:, c, :]]
            titles = ["FLAIR axial", "T1CE axial", "label axial", "FLAIR coronal"]
            for j, (v, t) in enumerate(zip(views, titles)):
                ax = axes[row, j]
                ax.imshow(np.rot90(v), cmap="gray" if j != 2 else "viridis", **({"vmin": 0, "vmax": 3} if j == 2 else {}))
                ax.set_title(f"{meta['dataset']}:{meta['case_id']}\n{t}", fontsize=7)
                ax.axis("off")
            row += 1
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"montage -> {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--png", type=Path)
    args = p.parse_args(argv)
    for d in args.dirs:
        print("\n".join(report(d)))
    if args.png:
        montage(args.dirs, args.png)


if __name__ == "__main__":
    main()
