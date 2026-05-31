# Hướng dẫn chạy Pipeline cũ (không dùng MedNeXt) + W&B

## 1. Chuẩn bị môi trường

```bash
# Vào thư mục project
cd /workspace/Counterfactual-Representation-Network

# Kích hoạt virtual environment
source ../.venv/bin/activate

# Cài đủ dependencies (nếu thiếu)
pip install scikit-learn pyyaml
```

> **Lưu ý**: Project này cần `PYTHONPATH=src` để Python tìm được module `crn`. Các script `run_full_pipeline.py` và `run_train_with_oom_retry.py` đã được fix sẵn để tự động thêm `src` vào path. Nếu chạy thủ công thì nhớ export:
> ```bash
> export PYTHONPATH=src
> ```

## 2. Cấu trúc Config

Các file config nằm trong thư mục:

```
configs/
├── crn_brats_segonly_unet_causal_contrastive_25d.yaml      # Pipeline cũ 2.5D chính
├── crn_brats_segonly_unet_causal_contrastive_25d_pilot.yaml # Pilot (nhanh, 1 epoch)
├── crn_brats_segonly_unet_causal.yaml                       # Pipeline cũ 2D U-Net
├── crn_brats_segonly_unet_causal_contrastive.yaml           # Pipeline cũ 2D + contrastive
├── crn_brats_segonly_unet_causal_regions.yaml               # Pipeline cũ + region losses
└── ...
```

**Các config KHÔNG dùng MedNeXt** sẽ có:
- `model.backbone_mode: 2.5d` (hoặc `2d`, `planar`, `3d`, `volumetric`)
- **KHÔNG có** `model.backbone_mode: mednext`

## 3. Các tham số quan trọng — thay ở đâu?

Mở file config YAML (vd: `configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml`), các tham số chính nằm ở các block sau:

### 3.1. Data
```yaml
data:
  train_csv: data/brats/brats_train.csv      # ← Đường dẫn train CSV
  val_csv: data/brats/brats_val.csv          # ← Đường dẫn validation CSV
  image_size: [128, 128]                     # ← Kích thước ảnh đầu vào
  in_channels: 4                             # ← Số channel (BraTS có 4 modality)
  slice_context: 5                           # ← Số slice lân cận (chỉ 2.5D)
  slice_context_layout: depth                # ← Cách xếp context: depth | channels
```

### 3.2. Model (Architecture)
```yaml
model:
  latent_dim: 128
  base_channels: 24                          # ← Channels đầu tiên của encoder
  num_seg_classes: 3                         # ← Số lớp segmentation (BraTS: ET, WT, TC)
  segmentation_head: unet
  backbone_mode: 2.5d                        # ← "2.5d" | "2d" | "3d" = pipeline cũ; "mednext" = pipeline mới
  norm_type: group
  group_norm_groups: 8
```

### 3.3. Loss weights
```yaml
loss:
  lambda_seg: 1.0                            # ← Trọng số segmentation loss
  lambda_region_adjustment: 0.35             # ← Trọng số causal adjustment
  lambda_region_cf_stability: 0.04           # ← Trọng số counterfactual stability
  lambda_region_cf_contrastive: 0.08         # ← Trọng số contrastive
  lambda_region_disease_swap: 0.10           # ← Trọng số disease swap
  lambda_dis: 0.01                           # ← Trọng số discriminator
  contrastive_temperature: 0.20              # ← Temperature cho contrastive
```

### 3.4. Training
```yaml
training:
  batch_size: 2                              # ← Batch size (sẽ bị nhân với batch-multiplier khi chạy script)
  epochs: 8                                  # ← Số epoch
  lr: 0.00003                                # ← Learning rate
  weight_decay: 0.0001                       # ← Weight decay
  num_workers: 2                             # ← Số worker cho DataLoader
  max_train_batches: 2048                    # ← Giới hạn batch train/epoch (bỏ đi nếu muốn train full)
  # max_val_batches: KHÔNG ĐƯỢC DÙNG với BraTS vì cần đủ slice 1 volume để tính volume metric!
  
  output_dir: runs/brats_segonly_unet_causal_contrastive_25d   # ← Thư mục lưu checkpoint
  device: auto                               # ← "cuda", "cpu", hoặc "auto"
  
  init_checkpoint: runs/.../best.pt          # ← Warm-start checkpoint (nếu không có sẽ tự bỏ qua)
  init_strict: false                         # ← Có strict load state_dict không
  
  checkpoint_metric: sweep_best_volume/brats/mean_dice   # ← Metric để chọn best checkpoint
  checkpoint_mode: max                       # ← "max" hoặc "min"
  checkpoint_threshold_sweep: 0.40,0.45,0.50,0.55,0.60,0.65  # ← Các threshold thử nghiệm
  
  loss_warmup:
    epochs: 3                                 # ← Số epoch warmup cho 1 số loss
    start_factor: 0.5                         # ← Factor khởi đầu
    keys:
      - lambda_dis
      - lambda_region_adjustment
      - ...
```

### 3.5. W&B (tùy chọn)
```yaml
wandb:
  project: counterfactual-representation-network
  entity: your-wandb-entity
  name: custom-run-name
  tags: [old-pipeline, brats]
```

Hoặc dùng biến môi trường trong file `.env`:
```bash
WANDB_API_KEY = your_key_here
WANDB_PROJECT=counterfactual-representation-network
WANDB_ENTITY=your-entity
```

## 4. Cách chạy Pipeline

### 4.1. Chạy nhanh (test/smoke)

Dùng config pilot hoặc tự tạo config rút gọn:

```bash
python scripts/run_full_pipeline.py \
  --config configs/crn_brats_segonly_unet_causal_contrastive_25d_pilot.yaml \
  --batch-multiplier 1 \
  --data-root data/brats20 \
  --output-dir data/brats
```

- `batch-multiplier=1`: giữ nguyên batch_size trong config (không nhân lên)
- Script tự động: chuẩn bị data (nếu thiếu CSV) → train → log W&B

### 4.2. Chạy full pipeline cũ

```bash
python scripts/run_full_pipeline.py \
  --config configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml \
  --batch-multiplier 1 \
  --data-root data/brats20 \
  --output-dir data/brats
```

> **Lưu ý với full run**:
> - Validation gồm ~74 volumes (~11.470 slices), mất ~20–30 phút cho val mỗi epoch
> - Tổng thời gian 8 epochs có thể mất vài giờ tùy GPU
> - Nếu muốn giảm thời gian: tạo val CSV nhỏ hơn (vd: chỉ 2–5 volumes) nhưng vẫn phải đủ **toàn bộ slice của mỗi volume** (không được cắt giữa chừng)

### 4.3. Chạy thuần (không qua pipeline script)

Nếu đã có data và muốn chạy trực tiếp:

```bash
export PYTHONPATH=src
export WANDB_API_KEY=your_key
export WANDB_PROJECT=counterfactual-representation-network

python -m crn.train --config configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml
```

### 4.4. Chạy với retry khi OOM

```bash
python scripts/run_train_with_oom_retry.py \
  --config configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml \
  --batch-multiplier 2 \
  --wandb-project counterfactual-representation-network
```

- Nếu OOM, script tự động giảm batch size xuống 1/2 và chạy lại
- Nếu `init_checkpoint` không tồn tại, script tự bỏ qua warm-start

## 5. Các file đã fix (lưu ý khi pull code mới)

Nếu reset/pull lại repo, cần áp lại các fix sau:

1. **`scripts/run_full_pipeline.py`** — thêm `PYTHONPATH=src` vào env của subprocess
2. **`scripts/run_train_with_oom_retry.py`** — thêm `sys.path.insert(0, str(repo_root / "src"))`
3. **Cài dependency**: `pip install scikit-learn`

## 6. Kiểm tra W&B đã lên chưa

Sau khi chạy, kiểm tra bằng API:

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

Hoặc mở link trực tiếp: `https://wandb.ai/<entity>/<project>/runs/<run_id>`

## 7. Ví dụ: Tạo config custom

Tạo file `configs/my_experiment.yaml`, copy từ config gốc rồi sửa:

```yaml
seed: 42

data:
  train_csv: data/brats/brats_train.csv
  val_csv: data/brats/brats_val.csv        # Hoặc tạo val nhỏ hơn để test nhanh
  image_size: [128, 128]
  in_channels: 4
  slice_context: 5
  slice_context_layout: depth

model:
  latent_dim: 128
  base_channels: 16                          # Giảm channels nếu GPU yếu
  num_seg_classes: 3
  segmentation_head: unet
  backbone_mode: 2.5d                        # Pipeline cũ
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
  epochs: 4                                   # Giảm epoch để test
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

Chạy:
```bash
python scripts/run_full_pipeline.py \
  --config configs/my_experiment.yaml \
  --batch-multiplier 1 \
  --data-root data/brats20 \
  --output-dir data/brats
```

## 8. Đánh giá (Evaluation)

Sau khi train xong, đánh giá bằng checkpoint tốt nhất:

```bash
export PYTHONPATH=src
python -m crn.evaluate \
  --checkpoint runs/my_experiment/best.pt \
  --split val \
  --batch-size 2 \
  --threshold-sweep 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
  --qualitative-count 4 \
  --counterfactual-metrics
```

## Tóm tắt

| Thao tác | File/Command |
|---------|-------------|
| **Chọn pipeline cũ** | Config có `backbone_mode: 2.5d` hoặc `2d` (không phải `mednext`) |
| **Thay đổi data path** | Block `data:` trong config YAML |
| **Thay đổi model** | Block `model:` trong config YAML |
| **Thay đổi loss weights** | Block `loss:` trong config YAML |
| **Thay đổi training params** | Block `training:` trong config YAML (epochs, lr, batch_size, output_dir, ...) |
| **Thay đổi W&B project/name** | Block `wandb:` trong config hoặc file `.env` |
| **Chạy train** | `python scripts/run_full_pipeline.py --config <config> --batch-multiplier 1` |
| **Chạy eval** | `PYTHONPATH=src python -m crn.evaluate --checkpoint <path>` |
