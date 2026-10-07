"""TRACE-Seg3D clean ISBI pipeline.

Modules
-------
preprocess  raw UTSW / BraTS -> harmonised .npz cases (crop on raw brain mask, cube pad, resize, z-score in brain)
splits      fixed patient-level train/val/test splits
data        torch datasets + augmentation
model       MedNeXt + feature-statistics context (AdaIN-style Counterfactual Context Transport)
losses      segmentation / CCT / proxy losses (single region map used everywhere)
metrics     Dice, HD95 in mm, structural prior, ET-empty aware reporting
train       training entry point (baseline | styleaug | trace)
calibrate   threshold + structural-prior selection on SOURCE validation only
evaluate    per-case evaluation, audit signals (CCT instability, entropy, TTA, ensemble)
audit       failure-detection metrics (Spearman, AUROC, AURC) for audit signals
summarize   mean +- std across seeds, paired Wilcoxon tests, LaTeX-ready tables
"""

__all__ = [
    "audit",
    "calibrate",
    "data",
    "evaluate",
    "losses",
    "metrics",
    "model",
    "preprocess",
    "splits",
    "summarize",
    "train",
]
