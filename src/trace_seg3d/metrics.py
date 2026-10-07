"""Per-case BraTS metrics with explicit empty-mask handling and HD95 in millimetres.

Conventions (state them in the paper):
* Dice: both prediction and reference empty -> 1.0 (BraTS convention); the per-case table also
  stores ``<R>_ref_empty`` so ET Dice can be reported on ET-present cases only.
* HD95: both empty -> 0 mm; exactly one empty -> 373.13 mm (BraTS penalty, image diagonal).
* Region masks come from the probabilistic-OR region map, thresholded per region, then the
  hierarchy ET <= TC <= WT is enforced by union (TC |= ET, WT |= TC).
* Structural prior: remove connected ET components smaller than ``min_et_voxels``
  (26-connectivity), then re-enforce the hierarchy.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label

REGIONS = ("WT", "TC", "ET")
HD95_EMPTY_PENALTY_MM = 373.13


def region_probs_np(prob: np.ndarray) -> np.ndarray:
    """[3(NCR,ED,ET),...] -> [3(WT,TC,ET),...] (probabilistic OR, same as losses)."""

    ncr, ed, et = prob[0], prob[1], prob[2]
    return np.stack([1 - (1 - ncr) * (1 - ed) * (1 - et), 1 - (1 - ncr) * (1 - et), et])


def region_targets_np(sub: np.ndarray) -> dict[str, np.ndarray]:
    sub = sub.astype(bool)
    return {"WT": sub[0] | sub[1] | sub[2], "TC": sub[0] | sub[2], "ET": sub[2]}


def enforce_hierarchy(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    et = masks["ET"].astype(bool)
    tc = masks["TC"].astype(bool) | et
    wt = masks["WT"].astype(bool) | tc
    return {"WT": wt, "TC": tc, "ET": et}


def threshold_regions(prob: np.ndarray, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    rp = region_probs_np(prob)
    return enforce_hierarchy({r: rp[i] >= float(thresholds[r]) for i, r in enumerate(REGIONS)})


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 0 or not mask.any():
        return mask
    lab, n = label(mask, structure=generate_binary_structure(3, 3))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[lab]


def structural_prior(masks: dict[str, np.ndarray], min_et_voxels: int) -> dict[str, np.ndarray]:
    out = dict(masks)
    out["ET"] = remove_small_components(masks["ET"], min_et_voxels)
    return enforce_hierarchy(out)


def hd95_mm(pred: np.ndarray, ref: np.ndarray, spacing: float | tuple[float, float, float]) -> float:
    pred, ref = pred.astype(bool), ref.astype(bool)
    if not pred.any() and not ref.any():
        return 0.0
    if not pred.any() or not ref.any():
        return HD95_EMPTY_PENALTY_MM
    sampling = (spacing,) * 3 if np.isscalar(spacing) else tuple(spacing)  # type: ignore[arg-type]
    st = generate_binary_structure(3, 1)
    ps = pred ^ binary_erosion(pred, st, border_value=0)
    rs = ref ^ binary_erosion(ref, st, border_value=0)
    d1 = distance_transform_edt(~rs, sampling=sampling)[ps]
    d2 = distance_transform_edt(~ps, sampling=sampling)[rs]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def case_metrics(pred: dict[str, np.ndarray], ref: dict[str, np.ndarray], spacing: float, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    voxel_ml = (spacing**3) / 1000.0
    for r in REGIONS:
        p, g = pred[r].astype(bool), ref[r].astype(bool)
        inter, ps, gs = float((p & g).sum()), float(p.sum()), float(g.sum())
        dice = 1.0 if ps + gs == 0 else 2 * inter / (ps + gs)
        out[f"{prefix}{r}_dice"] = dice
        out[f"{prefix}{r}_hd95"] = hd95_mm(p, g, spacing)
        out[f"{prefix}{r}_precision"] = float("nan") if ps == 0 else inter / ps
        out[f"{prefix}{r}_recall"] = float("nan") if gs == 0 else inter / gs
        out[f"{prefix}{r}_pred_ml"] = ps * voxel_ml
        out[f"{prefix}{r}_ref_empty"] = float(gs == 0)
        if r == "ET":
            lab, n = label(p, structure=generate_binary_structure(3, 3))
            fp_components = sum(1 for i in range(1, n + 1) if not (g & (lab == i)).any())
            out[f"{prefix}ET_fp_components"] = float(fp_components)
            out[f"{prefix}ET_fp_ml"] = float((p & ~g).sum()) * voxel_ml
    out[f"{prefix}mean_dice"] = float(np.mean([out[f"{prefix}{r}_dice"] for r in REGIONS]))
    out[f"{prefix}mean_hd95"] = float(np.mean([out[f"{prefix}{r}_hd95"] for r in REGIONS]))
    return out
