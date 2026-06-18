# Reproducing The Causal MedNeXt OOD Results

This document is the public-facing reproduction path for the current causal
MedNeXt OOD model. The strongest verified direction is UTSW -> BraTS.

## Current Best Smoke Result

```text
UTSW -> BraTS, 4 BraTS validation volumes
Mean Dice 0.7917269349
WT 0.8517204214
TC 0.8204188200
ET 0.7030415633
```

Expected artifact:

```text
runs/_ood_causal_adapt_brats_v5_et_precision_e2/brats_val_structural_min32_metrics.json
```

The row uses:

- ET-precision causal MedNeXt checkpoint
- support-aware SCM adjustment
- TC/ET-focused calibration
- anatomical structural prior with `min_component_size=32`

## Data

Expected local paths:

```text
data/brats/brats_train.csv
data/brats/brats_val.csv
data/brats/archive/BraTS2020_training_data/content/data/
data/brats/PKG - UTSW-Glioma/UTSW-Glioma/
data/brats/UTSW_Glioma_Metadata-2-1.tsv
```

Data is intentionally ignored by git.

## Re-run The Best Evaluation

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

## Reproduce The UTSW -> BraTS Refinement Path

This runner starts from the UTSW source checkpoint, performs the TC/ET-focused
BraTS adaptation, continues with causal SCM training, runs the ET-precision
continuation, and evaluates the structural prior.

```bash
SKIP_EXISTING=1 \
SUMMARY_PYTHON=python3 \
bash scripts/run_mednext_ood_et_focus_refine.sh
```

Set `SKIP_EXISTING=0` to regenerate checkpoints instead of reusing existing
artifacts.

## Full Source Checkpoints

UTSW source baseline:

```bash
PYTHON_BIN=.venv312_restore/bin/python \
OUT_DIR=runs/mednext_utsw_s_k3 \
BASELINE_EPOCHS=100 \
bash scripts/train_mednext_utsw_proxy_transport_baseline.sh
```

BraTS source baseline:

```bash
PYTHON_BIN=.venv312_restore/bin/python \
OUT_DIR=runs/mednext_brats_h5_s_k3_paper64 \
BASELINE_EPOCHS=50 \
CHECKPOINT_CALIBRATION_THRESHOLDS=0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
CHECKPOINT_CALIBRATION_OBJECTIVE=tc_et_min \
bash scripts/train_mednext_brats_paper64_baseline.sh
```

Do not set smoke variables such as `LIMIT_TRAIN_VOLUMES`,
`LIMIT_VAL_VOLUMES`, `MAX_TRAIN_BATCHES`, or `MAX_VAL_BATCHES` for full source
training.

## Source-only OOD Baselines

UTSW source -> BraTS target:

```bash
EXECUTE=1 MAX_VOLUMES=4 \
bash baselines/run_mednext_ood_baselines.sh mednext utsw-to-brats
```

BraTS source -> UTSW target:

```bash
EXECUTE=1 MAX_CASES=4 \
CHECKPOINT=runs/mednext_brats_h5_s_k3_paper64/best.pt \
bash baselines/run_mednext_ood_baselines.sh mednext brats-to-utsw
```

Use `MAX_CASES=0` to evaluate all UTSW validation cases.

## Qualitative Figures

```bash
PYTHONPATH=.:src .venv312_restore/bin/python scripts/make_mednext_structural_qual_figures.py \
  --checkpoint runs/_ood_causal_adapt_brats_v5_et_precision_e2/best.pt \
  --output-dir runs/figures/structural_prior_qual \
  --max-volumes 4 \
  --context-bank-size 4 \
  --max-context-bank-batches 2 \
  --adjustment-contexts 2 \
  --adjustment-context-selection diverse-nearest \
  --region-thresholds WT=0.65,TC=0.3,ET=0.55 \
  --structural-min-component-size 32 \
  --volume-size 64 \
  --num-workers 0
```

Expected summary:

```text
runs/figures/structural_prior_qual/summary.json
```

## Mechanism Ablation Rows

The current four-volume UTSW -> BraTS smoke showed:

| Mechanism | Mean Dice | WT | TC | ET |
|---|---:|---:|---:|---:|
| Zero-shot UTSW source | 0.299 | 0.433 | 0.271 | 0.192 |
| Few-shot adapted + calibration | 0.743 | 0.815 | 0.762 | 0.653 |
| SCM adjusted + calibration | 0.747 | 0.818 | 0.765 | 0.658 |
| ET-focused adapted + calibration | 0.784 | 0.845 | 0.812 | 0.694 |
| ET-precision SCM + calibration | 0.786 | 0.848 | 0.813 | 0.698 |
| ET-precision SCM + structural prior | 0.787 | 0.846 | 0.812 | 0.703 |
| ET-precision SCM + calibration + structural prior | 0.792 | 0.852 | 0.820 | 0.703 |

The structural prior should be reported alongside raw predictions. It encodes
the BraTS tumor hierarchy `ET <= TC <= WT` and removes implausible isolated
components without looking at the target mask.
