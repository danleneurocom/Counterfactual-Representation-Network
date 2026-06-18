# Causal MedNeXt

This folder contains the current main architecture for the OOD brain tumor
segmentation experiments.

## Files

```text
model.py                       3D MedNeXt encoder-decoder segmenter
causal.py                      Causal MedNeXt wrapper and context adjustment
common.py                      Shared losses, metrics, checkpoint, and TTA helpers
calibration.py                 WT/TC/ET threshold calibration utilities
dataset_cache.py               Optional disk cache wrapper for volume datasets
train_utsw.py                  UTSW source baseline training
train_causal_utsw.py           UTSW causal MedNeXt training
train_brats_h5.py              BraTS HDF5 source/adaptation baseline training
train_causal_brats_h5.py       BraTS HDF5 causal MedNeXt training/adaptation
evaluate_utsw.py               UTSW baseline evaluation
evaluate_causal_utsw.py        UTSW causal/SCM evaluation
evaluate_brats_h5.py           BraTS HDF5 baseline evaluation
evaluate_causal_brats_h5.py    BraTS HDF5 causal/SCM/structural evaluation
fit_plausibility_support.py    Optional support-fitting utility for plausibility studies
```

The model uses `src/crn/mednext_blocks.py` for MedNeXt block primitives and
`src/crn/metrics.py` for BraTS region and structural-prior metrics.

## Current Best OOD Evaluation

```bash
PYTHON_BIN=.venv312_restore/bin/python \
CHECKPOINT=runs/_ood_causal_adapt_brats_v5_et_precision_e2/best.pt \
OUT_JSON=runs/_ood_causal_adapt_brats_v5_et_precision_e2/brats_val_structural_min32_metrics.json \
CONTEXT_BANK_SIZE=4 \
CONTEXT_BANK_SAMPLING=farthest \
MAX_CONTEXT_BANK_BATCHES=2 \
MAX_VOLUMES=4 \
ADJUSTMENT_CONTEXTS=2 \
ADJUSTMENT_CONTEXT_SELECTION=diverse-nearest \
REGION_THRESHOLDS=WT=0.65,TC=0.3,ET=0.55 \
STRUCTURAL_PRIOR=1 \
STRUCTURAL_THRESHOLD=0.55 \
STRUCTURAL_MIN_COMPONENT_SIZE=32 \
SPLIT_NAME=ood_causal_v5_et_precision_structural_min32_smoke4 \
bash scripts/evaluate_mednext_brats_paper64_causal.sh
```

Expected smoke result:

```text
Mean Dice 0.792 | WT 0.852 | TC 0.820 | ET 0.703
```

## Reproduction

Use the top-level reproduction document for the full source/adaptation/SCM path:

```text
docs/REPRODUCING_MEDNEXT_OOD.md
```
