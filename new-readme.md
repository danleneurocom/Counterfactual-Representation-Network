# Initial Pipeline Run Guide (without MedNeXt) + W&B

## 1. Environment Preparation

```bash
# Go to project directory
cd /workspace/Counterfactual-Representation-Network

# Activate virtual environment
source ../.venv/bin/activate

# Install dependencies (if missing)
pip install scikit-learn pyyaml
```

> **Note**: This project requires `PYTHONPATH=src` for Python to find the `crn` module. The `run_full_pipeline.py` and `run_train_with_oom_retry.py` scripts have been fixed to automatically add `src` to the path. If running manually, remember to export:
> ```bash
> export PYTHONPATH=src
> ```

## 2. Config Structure

Config files are located in the directory:

```
configs/
├── crn_brats_segonly_unet_causal_contrastive_25d.yaml # Main 2.5D old pipeline
├── crn_brats_segonly_unet_causal_contrastive_25d_pilot.yaml # Pilot (fast, 1 epoch)
├── crn_brats_segonly_unet_causal.yaml # 2D U-Net old pipeline
├── crn_brats_segonly_unet_causal_contrastive.yaml # Old pipeline 2D + Contrastive
├── crn_brats_segonly_unet_causal_regions.yaml # Old Pipeline + Region Losses
└── ...
```
**Configurations that do NOT use MedNeXt** will have:
- `model.backbone_mode: 2.5d` (or `2d`, `planar`, `3d`, `volumetric`)
- **DO NOT have** `model.backbone_mode: mednext`

## 3. Important Parameters — Where to Change Them?

Open the YAML config file (e.g., `configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml`), the main parameters are located in the following blocks:

### 3.1. Data
```yaml
data:
train_csv: data/brats/brats_train.csv # ← CSV train path

val_csv: data/brats/brats_val.csv # ← CSV validation path
image_size: [128, 128] # ← Input image size

in_channels: 4 # ← Number of channels (BraTS has 4 modalitys)

slice_context: 5 # ← Number of adjacent slices (2.5D only)

slice_context_layout: depth # ← Context layout: depth | channels
```

### 3.2. Model (Architecture)
```yaml
model:

latent_dim: 128

base_channels: 24 # ← First encoder channels

num_seg_classes: 3 # ← Number of segmentation classes (BraTS: ET, WT, TC)

segmentation_head: unet

backbone_mode: 2.5d # ← "2.5d" | "2d" | "3d" = old pipeline; "mednext" = new pipeline

norm_type: group

group_norm_groups: 8
```

### 3.3. Losing weights
```yaml
loss: 
lambda_seg: 1.0 # ← Segmentation loss weight 
lambda_region_adjustment: 0.35 # ← Causal adjustment weight 
lambda_region_cf_stability: 0.04 # ← Counterfactual stability weight 
lambda_region_cf_contrastive: 0.08 # ← contrastive weight 
lambda_region_disease_swap: 0.10 # ← Disease swap weight 
lambda_dis: 0.01 # ← Discriminator weight 
contrastive_temperature: 0.20 # ← Temperature for contrastive
```

### 3.4. Training
```yaml
training:

batch_size: 2 # ← Batch size (will be multiplied by batch-multiplier when running the script)

epochs: 8 # ← Number of epochs

lr: 0.00003 # ← Learning rate

weight_decay: 0.0001 # ← Weight decay

num_workers: 2 # ← Number of workers for DataLoader

max_train_batches: 2048 # ← Batch train/epoch limit (remove if you want to train at full capacity)

# max_val_batches: NOT TO BE USED with BraTS because it requires a full volume slice to calculate the volume metric!

output_dir: runs/brats_segonly_unet_causal_contrastive_25d # ← Directory for storing checkpoints

device: auto # ← "cuda", "cpu", or "auto"

init_checkpoint: runs/.../best.pt # ← Warm-start checkpoint (if not present, it will be skipped)

init_strict: false # ← Is there a strict load state_dict?

checkpoint_metric: sweep_best_volume/brats/mean_dice # ← Metric to select the best checkpoint

checkpoint_mode: max # ← "max" or "min"

checkpoint_threshold_sweep: 0.40, 0.45, 0.50, 0.55, 0.60, 0.65 # ← Test thresholds

loss_warmup:

epochs: 3 # ← Number of warmup epochs for a loss value 
start_factor: 0.5 # ← Start factor 
keys: 
- lambda_dis 
- lambda_region_adjustment 
- ...
```

### 3.5. W&B (optional)
```yaml
wandb: 
project: counterfactual-representation-network 
entity: your-wandb-entity 
name: custom-run-name 
tags: [old-pipeline, brats]
```

Or use environment variables in the `.env` file:
```bash
WANDB_API_KEY = your_key_here
WANDB_PROJECT=counterfactual-representation-network
WANDB_ENTITY=your-entity
```

## 4. How to run Pipeline

### 4.1. Run fast (test/smoke)

Use pilot config or create your own shortened config:

```bash
python scripts/run_full_pipeline.py\ 
--config configs/crn_brats_segonly_unet_causal_contrastive_25d_pilot.yaml \ 
--batch-multiplier 1 \ 
--data-root data/brats20 \ 
--output-dir data/brats
```

- `batch-multiplier=1`: keeps the batch_size in the config (does not multiply)
- Automated script: prepares data (if CSV is missing) → trains → logs W&B

### 4.2. Run the initial full pipeline:

```bash
python scripts/run_full_pipeline.py \

--config configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml \

--batch-multiplier 1 \

--data-root data/brats20 \

--output-dir data/brats
```

> **Note for the full run**:

> - Validation consists of ~74 volumes (~11,470 slices), taking ~20–30 minutes per epoch.

> - The total time for 8 epochs can take several hours depending on the GPU.

> - To reduce the time: create smaller CSV validations (e.g., only 2–5 volumes) but still ensure **the entire slice of each volume** (no cutting in the middle).

### 4.3. Run plain (without a pipeline script)

If you already have data and want to run directly:

```bash
export PYTHONPATH=src
export WANDB_API_KEY=your_key
export WANDB_PROJECT=counterfactual-representation-network

python -m crn.train --config configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml
```

### 4.4. Run with retry when OOM occurs:

```bash
python scripts/run_train_with_oom_retry.py \

--config configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml \

--batch-multiplier 2 \

--wandb-project counterfactual-representation-network
```

- If OOM occurs, the script automatically reduces the batch size by half and reruns.

- If `init_checkpoint` does not exist, the script skips the warm-start.

## 5. Fixed files (note when pulling new code)

If resetting/pulling the repo, the following fixes need to be applied:

1. **`scripts/run_full_pipeline.py`** — add `PYTHONPATH=src` to the subprocess environment.

2. **`scripts/run_train_with_oom_retry.py`** — add `sys.path.insert(0, str(repo_root / "src"))`
3. **Install dependency**: `pip install scikit-learn`

## 6. Check if W&B is working

After running, check using the API:

```bash
source ../.venv/bin/activate
python -c "
import wandb
api = wandb.Api()
run = api.run('21522798-uit/counterfactual-representation-network/runs/RUN_ID')
print('State:', run.state)
print('Summary keys:', list(run.summary.keys())[:10])
"
```

Or open the link directly: `https://wandb.ai/<entity>/<project>/runs/<run_id>`

## 7. Example: Create a config custom

Create file `configs/my_experiment.yaml`, copy from original config and edit:

```yaml
seeds: 42

data: 
train_csv: data/brats/brats_train.csv 
val_csv: data/brats/brats_val.csv # Or create a smaller val for quick testing 
image_size: [128, 128] 
in_channels: 4 
slice_context: 5 
slice_context_layout: depth

model: 
latent_dim: 128 
base_channels: 16 # Reduce channels if GPU is weak 
num_seg_classes: 3 
segmentation_head: unet 
backbone_mode: 2.5d # Old Pipeline 
norm_type: group 
group_norm_groups: 8

loss: 
lambda_seg: 1.0 
lambda_region_adjustment: 0.35 
lambda_region_cf_stability: 0.04 
lambda_region_cf_contrastive: 0.08 
lambda_region_disease_swap: 0.10 
lambda_dis: 0.01 
contrastive_temperature: 0.20

training: 
batch_size: 2 
epochs: 4 # Reduce epochs for testing 
lr: 0.00003 
weight_decay: 0.0001 
num_workers: 2 
output_dir: runs/my_experiment 
device: auto 
checkpoint_metric: sweep_best_volume/brats/mean_dice 
checkpoint_mode: max 
checkpoint_threshold_sweep: 0.40,0.50,0.60 
loss_warmup: 
epochs: 2 
start_factor: 0.5 
keys: 
- lambda_dis 
- lambda_region_adjustment 
- lambda_region_cf_stability 
- lambda_region_cf_contrastive 
- lambda_region_disease_swap
```

Run:
```bash
python scripts/run_full_pipeline.py\ 
--config configs/my_experiment.yaml \ 
--batch-multiplier 1 \ 
--data-root data/brats20 \ 
--output-dir data/brats
```

## 8. Evaluation

After training is complete, evaluate with the best checkpoint:

```bash
export PYTHONPATH=src
python -m crn.evaluate\ 
--checkpoint runs/my_experiment/best.pt \ 
--split val \ 
--batch-size 2 \ 
--threshold-sweep 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65

--qualitative-count 4

--counterfactual-metrics

## Summary

| Operation | File/Command |

|---------|-------------|

| **Select old pipeline** | Config with `backbone_mode: 2.5d` or `2d` (not `mednext`) |

| **Change data path** | Block `data:` in YAML config |

| **Change model** | Block `model:` in YAML config |

| **Change loss weights** | Block `loss:` in YAML config |

| **Change training params** | Block `training:` in YAML config (epochs, lr, batch_size, output_dir, ...) |

| **Change W&B project/name** | Block `wandb:` in config or `.env` file |
| **Running train** | `python scripts/run_full_pipeline.py --config <config> --batch-multiplier 1` |
| **Run eval** | `PYTHONPATH=src python -m crn.evaluate --checkpoint <path>` |
