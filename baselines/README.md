# Baseline Comparison Models

This folder contains local baseline implementations and a harness for related
external baselines used to compare against this repository's causal MedNeXt
segmentation model.

## Dataset Policy

All reported baseline comparisons must run on this repository's datasets:

- `brats`: BraTS2020 HDF5 slices under `data/brats`
- `utsw`: UTSW glioma NIfTI cases under `data/brats/PKG - UTSW-Glioma`
- `brisc`: BRISC2025 2D image/mask data under `data/brats/brisc2025`

Do not report numbers from an upstream paper repo's original dataset split as a
baseline for this project. External paper repos are literature/code references
unless they are run through the adapted local-data runner in
[related_work/adapted_baselines.py](related_work/adapted_baselines.py).

## Local Baselines

These baselines have local code in this repository and can be trained with this
project's scripts.

| Model | Venue | Paper | Upstream code | Local code | Status |
|---|---|---|---|---|---|
| MedNeXt | MICCAI 2023 | [Paper](https://conferences.miccai.org/2023/papers/410-Paper1656.html) | [GitHub](https://github.com/MIC-DKFZ/MedNeXt) | [baselines/mednext](mednext/) | Native on BraTS/UTSW; adapted 2D BRISC runner available |
| SegFormer3D | CVPRW 2024 | [Paper](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/papers/Perera_SegFormer3D_An_Efficient_Transformer_for_3D_Medical_Image_Segmentation_CVPRW_2024_paper.pdf) | [GitHub](https://github.com/OSUPCVLab/SegFormer3D) | [baselines/segformer3d](segformer3d/) | Runnable on BraTS and UTSW; BRISC adapter pending |

## Counterfactual Context Transport Evaluation

The causal MedNeXt UTSW evaluator can test Counterfactual Context Transport
(CCT) as an inference-time segmentation mechanism. CCT holds the disease latent
fixed, transports it across a proxy-context bank, and reports both an
interventional consensus mask and an instability-gated mask.

```bash
PYTHON_BIN=.venv312_restore/bin/python \
CHECKPOINT=runs/mednext_utsw_causal_proxy/best.pt \
OUT_JSON=runs/mednext_utsw_causal_proxy/utsw_val_cct_metrics.json \
CCT_CONTEXTS=8 CONTEXT_BANK_SIZE=64 CONTEXT_BANK_SAMPLING=farthest \
bash scripts/evaluate_mednext_utsw_proxy_transport_causal.sh
```

The main reported CCT keys are `cct_consensus/brats/mean_dice`,
`cct_stability_gated/brats/mean_dice`, and the two
`intervention/cct_*_minus_factual_mean_dice` deltas.

## UTSW to BraTS OOD Smoke

Use the UTSW-to-BraTS smoke runner to reproduce the current OOD path from the
best UTSW checkpoint, through TC/ET-weighted BraTS adaptation, then the causal
SCM continuation:

```bash
bash scripts/run_mednext_ood_utsw_to_brats_smoke.sh
```

By default the runner skips existing artifacts, so it can safely summarize the
current best small-sample run while the full UTSW job is still active. To also
evaluate Counterfactual Context Transport on the causal checkpoint:

```bash
RUN_CCT_EVAL=1 CCT_SELECTION=diverse-nearest \
bash scripts/run_mednext_ood_utsw_to_brats_smoke.sh
```

The CCT selector can be `uniform`, `nearest`, `farthest`, or
`diverse-nearest`. `uniform` estimates a context marginal; `diverse-nearest`
is the OOD-focused setting because it keeps transported contexts on the target
support while still testing context variation.

The same selection policy is also available for SCM adjusted logits during
causal training/evaluation:

```bash
ADJUSTMENT_CONTEXT_SELECTION=diverse-nearest \
bash scripts/run_mednext_ood_utsw_to_brats_smoke.sh
```

Keep `ADJUSTMENT_CONTEXT_SELECTION=uniform` when reproducing the existing
small-sample numbers exactly; use `diverse-nearest` for the next OOD mechanism
ablation aimed at improving TC/ET stability.

To reproduce the current strongest TC/ET-focused refinement and its fine
threshold sweep:

```bash
SKIP_EXISTING=1 bash scripts/run_mednext_ood_et_focus_refine.sh
```

Current verified UTSW-to-BraTS smoke evidence:

| Stage | Mean Dice | WT | TC | ET |
|---|---:|---:|---:|---:|
| Zero-shot UTSW best on BraTS, 2 val volumes | 0.299 | 0.433 | 0.271 | 0.192 |
| Few-shot adapted, 4 val volumes, calibrated | 0.743 | 0.815 | 0.762 | 0.653 |
| Causal SCM adjusted, 4 val volumes, calibrated | 0.747 | 0.818 | 0.765 | 0.658 |
| ET-focused adapted continuation, 4 val volumes, calibrated | 0.784 | 0.845 | 0.812 | 0.694 |
| ET-focused support-aware SCM, 4 val volumes, calibrated | 0.785 | 0.846 | 0.813 | 0.697 |
| ET-focused support-aware SCM, 4 val volumes, fine calibrated | 0.786 | 0.847 | 0.813 | 0.697 |
| ET-precision support-aware SCM, 4 val volumes, TC/ET calibrated | 0.786 | 0.848 | 0.814 | 0.698 |
| ET-precision support-aware SCM + structural prior, 4 val volumes | 0.792 | 0.852 | 0.820 | 0.703 |

This is the valid OOD framing for the paper: the zero-shot row measures the
cross-dataset domain intervention, while the adapted and SCM rows measure
recoverability and context-adjusted stability under that shift.

## OOD Baseline Entry Point

For source-only baseline comparisons, use the baseline-folder runner. It is a
dry-run by default; set `EXECUTE=1` to actually run. This keeps baseline OOD
evaluation separate from our causal SCM runs while using the same local datasets
and metric code.

UTSW source -> BraTS target:

```bash
bash baselines/run_mednext_ood_baselines.sh mednext utsw-to-brats

EXECUTE=1 MAX_VOLUMES=4 \
bash baselines/run_mednext_ood_baselines.sh mednext utsw-to-brats
```

BraTS source -> UTSW target:

```bash
bash baselines/run_mednext_ood_baselines.sh mednext brats-to-utsw

EXECUTE=1 MAX_CASES=4 \
CHECKPOINT=runs/mednext_brats_h5_s_k3_paper64/best.pt \
bash baselines/run_mednext_ood_baselines.sh mednext brats-to-utsw
```

Use `MAX_CASES=0` for all UTSW validation cases. The BraTS->UTSW path pins UTSW
case ids from `runs/mednext_utsw_s_k3/splits.json` because a BraTS checkpoint's
own split file contains BraTS volume ids, not UTSW case ids.

## Related-Work Baselines

These are the ten paper baselines selected for comparison in
[literature/related_work_causal_efficient_segmentation.md](../literature/related_work_causal_efficient_segmentation.md).
They have public upstream code, but most are not plug-compatible with this
repo's BraTS, BRISC, or UTSW loaders. The adapted local runner provides small
epoch local-data baselines for all ten slugs. Use
[related_work/run_baseline.sh](related_work/run_baseline.sh) only to inspect or
clone upstream code. See [related_work/PORTING_PLAN.md](related_work/PORTING_PLAN.md)
for fidelity notes.

| Slug | Model | Venue | Paper | Public code | Local runner/status |
|---|---|---|---|---|---|
| `csdg` | Causality-inspired Single-source Domain Generalization for Medical Image Segmentation | IEEE TMI 2022 | [Paper](https://ieeexplore.ieee.org/document/9961940) | [GitHub](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization) | Adapted local runner on BraTS/UTSW/BRISC |
| `caussl` | CauSSL: Causality-inspired Semi-supervised Learning for Medical Image Segmentation | ICCV 2023 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Miao_CauSSL_Causality-inspired_Semi-supervised_Learning_for_Medical_Image_Segmentation_ICCV_2023_paper.html) | [GitHub](https://github.com/JuzhengMiao/CauSSL) | Adapted local runner on BraTS/UTSW/BRISC |
| `causalclipseg` | CausalCLIPSeg: Unlocking CLIP's Potential in Referring Medical Image Segmentation with Causal Intervention | MICCAI 2024 | [Paper](https://papers.miccai.org/miccai-2024/124-Paper3127.html) | [GitHub](https://github.com/WUTCM-Lab/CausalCLIPSeg) | Adapted local runner on BraTS/UTSW/BRISC |
| `icmseg` | Generalizable Single-Source Cross-Modality Medical Image Segmentation via Invariant Causal Mechanisms | WACV 2025 | [Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Chen_Generalizable_Single-Source_Cross-Modality_Medical_Image_Segmentation_via_Invariant_Causal_Mechanisms_WACV_2025_paper.pdf) | [GitHub](https://github.com/ratschlab/ICMSeg) | Adapted local runner on BraTS/UTSW/BRISC |
| `ciseg` | CiSeg: Unsupervised Cross-Modality Adaptation for 3D Medical Image Segmentation via Causal Intervention | IEEE TMI 2025/2026 | [Paper](https://pubmed.ncbi.nlm.nih.gov/41082440/) | [GitHub](https://github.com/lvpeiqing/CiSeg) | Adapted local runner on BraTS/UTSW/BRISC |
| `causalad` | CausalAD: Collaborative Learning of Augmentation and Disentanglement for Semi-Supervised Domain Generalized Medical Image Segmentation | IEEE TMI 2025/2026 | [Paper](https://ieeexplore.ieee.org/document/11115113) | [GitHub](https://github.com/Senyh/CausalAD) | Adapted local runner on BraTS/UTSW/BRISC |
| `cauaug` | CauAug: Causality-Adjusted Data Augmentation for Domain Continual Medical Image Segmentation | IEEE JBHI 2025/2026 | [Paper](https://pubmed.ncbi.nlm.nih.gov/40577309/) | [GitHub](https://github.com/PerceptionComputingLab/CauAug_DCMIS) | Adapted local runner on BraTS/UTSW/BRISC |
| `cfseg` | CF-Seg: Counterfactuals Meet Segmentation | MICCAI 2025 | [Paper](https://papers.miccai.org/miccai-2025/0144-Paper4597.html) | [GitHub](https://github.com/biomedia-mira/CF-Seg) | Adapted local runner on BraTS/UTSW/BRISC |
| `mednext` | MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation | MICCAI 2023 | [Paper](https://conferences.miccai.org/2023/papers/410-Paper1656.html) | [GitHub](https://github.com/MIC-DKFZ/MedNeXt) | Local runnable via this repo |
| `dmfnet` | 3D Dilated Multi-Fiber Network for Real-Time Brain Tumor Segmentation in MRI | MICCAI 2019 | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_21) | [GitHub](https://github.com/China-LiuXiaopeng/BraTS-DMFNet) | Adapted local runner on BraTS/UTSW/BRISC |

## Local-Dataset Commands

The checked-in local runners currently live in `scripts/` for MedNeXt and in
`baselines/segformer3d/` for SegFormer3D. Use the OOD baseline entry point above
for source-only cross-dataset MedNeXt comparisons. Related-work adapters should
be added here only after they have local-data loaders and the same OOD protocol.
