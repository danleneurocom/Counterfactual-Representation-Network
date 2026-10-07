"""Qualitative figure from evaluate.py ``--save-maps`` outputs: FLAIR, GT, prediction, CCT instability.

    python -m trace_seg3d.figures runs/isbi/utsw/trace_s0/eval_brats_test/maps_*.npz --out fig_cct.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np



def _label_rgb(sub: np.ndarray) -> np.ndarray:
    """sub [3(NCR,ED,ET),H,W] -> RGBA overlay (ED green, NCR blue, ET yellow)."""

    rgba = np.zeros((*sub.shape[1:], 4))
    rgba[sub[1] > 0.5] = (0.2, 0.8, 0.3, 0.6)
    rgba[sub[0] > 0.5] = (0.2, 0.4, 1.0, 0.7)
    rgba[sub[2] > 0.5] = (1.0, 0.9, 0.1, 0.8)
    return rgba


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("maps", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)
    n = len(args.maps)
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.4 * n), squeeze=False)
    for i, path in enumerate(args.maps):
        d = np.load(path)
        img, tgt = d["image"].astype(np.float32), d["target"]
        prob, u = d["prob"].astype(np.float32), d["instability"].astype(np.float32)
        z = int(np.argmax(tgt.max(0).sum(axis=(0, 1)))) if tgt.any() else img.shape[-1] // 2
        flair = np.rot90(img[0][:, :, z])
        pred = (prob[:, :, :, z] >= args.threshold).astype(np.float32)
        u_slice = u.max(0)[:, :, z]
        panels = [("FLAIR", None), ("GT", _label_rgb(tgt[:, :, :, z])), ("Prediction", _label_rgb(pred)), ("CCT instability", None)]
        for j, (title, overlay) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(flair, cmap="gray")
            if overlay is not None:
                ax.imshow(np.rot90(overlay))
            if j == 3:
                im = ax.imshow(np.rot90(u_slice), cmap="magma", alpha=0.85, vmin=0, vmax=max(0.05, float(u_slice.max())))
                fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{path.stem.removeprefix('maps_')} – {title}" if j == 0 else title, fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
