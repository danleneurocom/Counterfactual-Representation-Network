# Counterfactual Representation Network

Research scaffold for **Counterfactual Representation Network for Fair and Interpretable Imaging**.

The initial draft treated context/confounders as factors to remove from prediction. The corrected framing here follows the supervisor feedback: confounders can also be causal parents of the label, so prediction should condition on both latent components while estimating the intervention effect of the disease factor.

Core model:

```text
z_d = E_d(x)
z_c = E_c(x)
y_hat = f_cls(z_d, z_c)
m_hat = f_seg(z_d, z_c)
x_hat = G(z_d, z_c)
```

Core training idea:

```text
p(y | do(z_d)) ~= E_{z_c}[p(y | z_d, z_c)]
```

In practice, `crn.losses.backdoor_adjusted_logits` averages predictions over context latents from the minibatch. Context swapping is used as a bounded stability regularizer, not a strict invariance constraint.

## Project Layout

```text
configs/                  Example CheXpert and BraTS configs
docs/                     Revised method and experiment plan
src/crn/                  Model, losses, data loading, training entry point
tests/                    Focused tests for the causal loss utilities
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev]"
```

Install a PyTorch build appropriate for your machine if the default dependency resolver does not choose one:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data Format

The training script expects CSV files. Minimal classification CSV:

```csv
path,Cardiomegaly,Edema,No Finding
patient001.png,0,1,0
patient002.png,0,0,1
```

Minimal segmentation CSV:

```csv
path,mask,tumor_present
case001.npy,case001_mask.npy,1
case002.npy,case002_mask.npy,0
```

Supported image inputs are common PIL-readable images and `.npy` arrays. `.npy` arrays may be `H x W`, `H x W x C`, or `C x H x W`.

For the Kaggle-style BraTS2020 HDF5 slices, prepare volume-level train/validation splits with:

```bash
python scripts/prepare_brats_metadata.py \
  --metadata "BraTS20 Training Metadata.csv" \
  --data-root /Users/lenguyenlinhdan/Downloads/content/data \
  --output-dir data/brats \
  --require-files
```

This writes `data/brats/brats_train.csv` and `data/brats/brats_val.csv`. The loader expects each `.h5` file to contain `image` and `mask` keys; the current BraTS configs use `target` as the binary slice label and preserve the 3-channel tumor subregions for segmentation.

## Run

Edit the data paths in a config, then run:

```bash
PYTHONPATH=src python -m crn.train --config configs/crn_chexpert.yaml
```

or:

```bash
PYTHONPATH=src python -m crn.train --config configs/crn_brats.yaml
```

Outputs are written to the config's `training.output_dir`.

For the current strongest BraTS segmentation recipe, use:

```bash
PYTHONPATH=src python -m crn.train --config configs/crn_brats_segonly_unet_causal.yaml
```

This is the mainline causal-region setup. The `*_memory*.yaml` configs remain available for counterfactual-memory experiments, but they are experimental and have not beaten the mainline model yet.

To try the lesion-aware counterfactual contrastive upgrade on top of the strong causal-region checkpoint, use:

```bash
PYTHONPATH=src python -m crn.train --config configs/crn_brats_segonly_unet_causal_contrastive.yaml
```

This adds a lesion-aware counterfactual contrastive loss with:

- anchor: factual lesion-conditioned segmentation features
- positive: backdoor-adjusted same-disease features
- hard negative: matched disease-swapped counterfactual features

For a quick BraTS sanity check before full training:

```bash
PYTHONPATH=src python -m crn.train --config configs/crn_brats_smoke.yaml
```

To evaluate a saved checkpoint:

```bash
PYTHONPATH=src python -m crn.evaluate --checkpoint runs/brats_crn_multiregion/best.pt --split val --batch-size 8
```

This writes a metrics JSON next to the checkpoint, for example `runs/brats_crn_multiregion/best_val_metrics.json`.

To run volume-level evaluation with a threshold sweep and export qualitative figures:

```bash
PYTHONPATH=src python -m crn.evaluate \
  --checkpoint runs/brats_crn_multiregion/best.pt \
  --split val \
  --batch-size 8 \
  --threshold-sweep 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
  --qualitative-count 4
```

This additionally writes:

- `runs/brats_crn_multiregion/best_val_threshold_sweep.json`
- `runs/brats_crn_multiregion/best_val_qualitative/`

The qualitative export saves representative best and worst validation volumes as PNG overlays together with a `summary.json` index.
It now also exports causal counterfactual image panels for representative volumes:

- factual reconstruction
- context-swapped counterfactual image and segmentation
- disease-swapped counterfactual image and segmentation
- a causal disease `do`-effect map
- a context-shift sensitivity map

Volume-level evaluation now reports BraTS-style Dice and HD95 for `WT`, `TC`, and `ET`, plus per-subregion HD95.

To independently tune `WT`, `TC`, and `ET` thresholds with simple 3D post-processing:

```bash
PYTHONPATH=src python -m crn.evaluate \
  --checkpoint runs/brats_crn_multiregion/best.pt \
  --split val \
  --batch-size 8 \
  --threshold-sweep 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
  --tune-brats-regions \
  --qualitative-count 4
```

This also writes `runs/brats_crn_multiregion/best_val_region_tuning.json` with the selected region-wise thresholds and cleanup settings.

The current BraTS configs are now set up for 3-channel subregion segmentation with BraTS region metrics:

- `ncr_net`
- `edema`
- `enhancing_tumor`

and evaluation reports official derived regions:

- `WT = ncr_net | edema | enhancing_tumor`
- `TC = ncr_net | enhancing_tumor`
- `ET = enhancing_tumor`
