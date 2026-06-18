# Causal MedNeXt for OOD Brain Tumor Segmentation

This repository contains the code used for our current causal MedNeXt
out-of-distribution brain tumor segmentation experiments. The main model is a
3D MedNeXt segmenter augmented with a structural causal mechanism for
context-aware segmentation, TC/ET-aware calibration, and anatomical
region-plausibility constraints.

The current strongest verified OOD smoke result is UTSW -> BraTS:

```text
Mean Dice 0.792 | WT 0.852 | TC 0.820 | ET 0.703
```

The corresponding evaluation artifact is:

```text
runs/_ood_causal_adapt_brats_v5_et_precision_e2/brats_val_structural_min32_metrics.json
```

Large checkpoints, run folders, and medical imaging data are intentionally not
tracked in git. Share checkpoints separately as artifacts.

## Repository Map

```text
src/causal_mednext/       Canonical Causal MedNeXt model package
src/causal_mednext/backbone.py
                           3D MedNeXt encoder-decoder backbone
src/causal_mednext/causal_model.py
                           SCM-style disease/context model and adjustment logic
src/causal_mednext/mechanisms/
                           Calibration, structural prior, hierarchy, and veto mechanisms
baselines/mednext/        Train/eval wrappers and compatibility imports for scripts
baselines/segformer3d/    Local SegFormer3D baseline and shared UTSW/BraTS loaders
baselines/README.md       Baseline comparison and OOD baseline entry points
scripts/                  Reproducible train/eval wrappers for the MedNeXt OOD path
src/crn/metrics.py        BraTS metrics and structural-prior evaluation utilities
tests/                    Focused regression tests for metrics and MedNeXt plumbing
docs/                     Reproduction notes and public-release checklist
```

The legacy CRN prototype code is kept only where it supports the current
MedNeXt path. The public mainline model is `causal_mednext`; the
`baselines/mednext` package keeps the original experiment commands stable.

## Setup

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the PyTorch build appropriate for your machine if needed:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use `PYTHONPATH=.:src` for direct script execution.

## Data Layout

The scripts expect local datasets in these locations:

```text
data/brats/brats_train.csv
data/brats/brats_val.csv
data/brats/archive/BraTS2020_training_data/content/data/
data/brats/PKG - UTSW-Glioma/UTSW-Glioma/
data/brats/UTSW_Glioma_Metadata-2-1.tsv
```

These files are ignored by git because they are large and/or dataset controlled.

If you need to regenerate the BraTS HDF5 CSVs:

```bash
python scripts/prepare_brats_metadata.py \
  --metadata "BraTS20 Training Metadata.csv" \
  --data-root /path/to/BraTS2020_training_data/content/data \
  --output-dir data/brats \
  --require-files
```

## Reproduce The Current Best OOD Path

The best verified OOD direction is UTSW -> BraTS. To summarize or reproduce the
current small-sample OOD path:

```bash
SKIP_EXISTING=1 \
SUMMARY_PYTHON=python3 \
bash scripts/run_mednext_ood_et_focus_refine.sh
```

To evaluate the current best checkpoint with TC/ET calibration and the
structural prior:

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

See [docs/REPRODUCING_MEDNEXT_OOD.md](docs/REPRODUCING_MEDNEXT_OOD.md) for the
full source, adaptation, causal continuation, baseline, and reverse-direction
commands.

## Baseline OOD Comparisons

Source-only MedNeXt OOD baselines live under `baselines/`:

```bash
bash baselines/run_mednext_ood_baselines.sh list

EXECUTE=1 MAX_VOLUMES=4 \
bash baselines/run_mednext_ood_baselines.sh mednext utsw-to-brats

EXECUTE=1 MAX_CASES=4 \
CHECKPOINT=runs/mednext_brats_h5_s_k3_paper64/best.pt \
bash baselines/run_mednext_ood_baselines.sh mednext brats-to-utsw
```

## Qualitative Figures

Generate structural-prior qualitative panels:

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

## Tests

Focused checks for the public mainline:

```bash
PYTHONPATH=.:src python -m py_compile \
  baselines/mednext/causal.py \
  baselines/mednext/evaluate_causal_brats_h5.py \
  baselines/mednext/evaluate_causal_utsw.py \
  scripts/make_mednext_structural_qual_figures.py

PYTHONPATH=.:src python -m pytest tests/test_metrics.py tests/test_mednext_baseline.py
```

## Artifact Policy

Do not commit:

- `data/`
- `runs/`
- `experiments/`
- virtual environments
- model checkpoints (`*.pt`, `*.ckpt`)
- private research notes

For public release cleanup, see
[docs/PUBLIC_RELEASE_CHECKLIST.md](docs/PUBLIC_RELEASE_CHECKLIST.md).
