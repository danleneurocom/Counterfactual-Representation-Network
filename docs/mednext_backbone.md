# MedNeXt 3D Backbone

This document describes how to use the **MedNeXt 3D** backbone in the Counterfactual Representation Network.

## Supported Backbones

The `model.backbone_mode` config key controls the encoder architecture:

| Value | Alias | Encoder | Decoder Segmentation Head |
|-------|-------|---------|---------------------------|
| `"2d"` | `"planar"` | `ImageEncoder` (2D CNN) | `SpatialDecoder` (latent) or `ContextualUNetDecoder` (unet) |
| `"volumetric"` | `"3d"`, `"2.5d"`, `"25d"` | `VolumetricImageEncoder` (3D CNN) | `VolumetricContextualUNetDecoder` (unet) |
| `"mednext"` | `"mednext3d"`, `"mednext_3d"` | `MedNeXtEncoder` (MedNeXt 3D) | `VolumetricContextualUNetDecoder` (unet) |

## MedNeXt-Specific Parameters

In the `model:` block of your YAML config:

```yaml
model:
  backbone_mode: mednext          # required
  mednext_variant: L              # S | B | M | L  (default L)
  mednext_kernel_size: 5          # 3 | 5 | 7      (default 5)
  base_channels: 32               # overrides variant default n_channels
```

### Variant Presets

| Variant | `n_channels` | Blocks per stage | Exp ratios |
|---------|-------------|------------------|------------|
| S | 32 | [2, 2, 2, 2] | [2, 3, 4, 4] |
| B | 32 | [2, 2, 2, 2] | [2, 3, 4, 4] |
| M | 32 | [3, 4, 4, 4] | [2, 3, 4, 4] |
| L | 32 | [3, 4, 8, 8] | [3, 4, 8, 8] |

The encoder channel progression is always `[base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*8]`.

## Data Requirements for 3D Backbones

MedNeXt (and any volumetric backbone) expects **3D input**: `(B, C, D, H, W)`.

Use `slice_context > 1` with `slice_context_layout: depth` in the data config so that neighbouring axial slices are stacked along the depth dimension:

```yaml
data:
  slice_context: 5                # odd positive integer
  slice_context_layout: depth     # stacks slices into 3D volume
```

Your CSV must contain `volume` and `slice` columns so the loader knows how to group and order context slices.

## Segmentation Head

For volumetric backbones you **must** set:

```yaml
model:
  segmentation_head: unet
```

The `VolumetricContextualUNetDecoder` uses the encoder's multi-resolution feature maps (skip connections) and returns the **center slice** segmentation logits. This matches the standard 2.5D / volumetric BraTS practice where the model sees a small depth context but predicts the middle slice.

## Reconstruction Loss

The reconstructor (`SpatialDecoder`) always outputs a 2D image. When the input is 3D, the reconstruction loss automatically extracts the **center slice** from the ground-truth volume before computing L1 loss. No config change is needed.

## Quick Examples

### Smoke-test MedNeXt-S (1 epoch, tiny data)

```bash
python -m crn.train --config configs/crn_brats_smoke_mednext3d.yaml
```

### Smoke-test MedNeXt-L with full causal losses

```bash
python -m crn.train --config configs/crn_brats_smoke_mednextL_full.yaml
```

### Full pipeline (train + eval)

```bash
python scripts/run_pipeline.py --config configs/crn_brats_smoke_mednext3d.yaml --wandb disabled
```

## Evaluation

Eval works out of the box with any backbone. For volume-level metrics and threshold sweep:

```bash
python -m crn.evaluate \
  --checkpoint runs/brats_smoke_mednextL_full/best.pt \
  --split val \
  --threshold-sweep 0.40,0.45,0.50,0.55,0.60,0.65 \
  --qualitative-count 4
```

## Warmstarting Across Backbones

The training script supports loading a checkpoint trained with a different backbone via `_load_compatible_state_dict`. The loader will:

1. Skip keys that do not exist in the new model.
2. Inflate 2D conv kernels to 3D by repeating along depth when moving from a 2D to a 3D backbone.
3. Report counts of missing / unexpected / adapted keys.

Use `init_checkpoint: /path/to/checkpoint.pt` and `init_strict: false` in the training config.

## Known Limitations

- The current `MedNeXtEncoder` implements the **encoder stages** only. It does not include the deep bottleneck blocks described in the full MedNeXt paper (those would sit between encoder and decoder). This is sufficient for the CRN segmentation pipeline because the UNet decoder uses the encoder feature maps directly.
- `max_val_batches` is **incompatible** with volume-level evaluation (data with `volume`/`slice` columns). Either remove `max_val_batches` from the training config or disable volume metrics.
