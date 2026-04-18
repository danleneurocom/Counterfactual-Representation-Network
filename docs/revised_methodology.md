# Revised Methodology

## Causal Correction

The original proposal treated the context latent `z_c` as information that should be removed from inference, and relied on `z_d` alone for prediction. The supervisor feedback correctly points out the causal issue: under confounding, context variables are often also causes of the target. Removing them from inference can therefore remove genuine causal parents of the output.

The corrected framing is **causal effect estimation under latent confounding**, not confounder removal.

## Current Model

For an input image `x`, the current framework learns:

```text
z_d = E_d(x)
z_c = E_c(x)
m_hat = f_seg(z_d, z_c)
x_hat = G(z_d, z_c)
```

In the current BraTS setup:

- `E_d` and `E_c` are separate encoders
- `f_seg` is a context-conditioned U-Net decoder
- the decoder uses disease skip features together with a latent head input formed from both `z_d` and `z_c`
- `G` is a reconstruction decoder used for counterfactual interpretability

The model is therefore explicitly **confounder-aware**: segmentation depends on both latent components, rather than forcing strict invariance to context.

## Latent Backdoor Adjustment for Segmentation

The estimand of interest is:

```text
p(m | do(z_d))
```

The implementation approximates this by averaging segmentation predictions over a small bank of context latents:

```text
p(m | do(z_d_i)) ~= (1 / K) sum_k p(m | z_d_i, z_c_k)
```

This preserves the role of `z_c` in the segmentation model while estimating the interventional contribution of the disease factor. In the current mainline training recipe, the adjustment is applied directly to segmentation logits and then supervised at the BraTS region level.

## Counterfactual Mechanisms

### Context Counterfactuals

Context swaps keep `z_d` fixed and replace `z_c` with a matched alternative context:

```text
m_cf_ctx = f_seg(z_d, z_c')
```

We do **not** enforce strict invariance. Instead, we use a bounded stability constraint:

```text
max(|m_hat - m_cf_ctx| - margin, 0)
```

This allows small context effects while penalizing unstable context sensitivity.

### Disease Counterfactuals

Disease swaps keep context fixed and replace the disease factor with a matched donor disease latent:

```text
m_cf_dis = f_seg(z_d', z_c)
```

This approximates a disease intervention under fixed context, and is supervised using the donor lesion structure.

## BraTS Region-Aware Causal Supervision

The current method is optimized for BraTS region outputs:

- `WT = NCR union ED union ET`
- `TC = NCR union ET`
- `ET = ET`

The causal losses are therefore applied at the region level, not just the raw channel level. The current mainline loss combines:

```text
L_total =
    lambda_seg * L_seg
  + lambda_region_adjustment * L_region_adjustment
  + lambda_region_cf_stability * L_region_context_stability
  + lambda_region_disease_swap * L_region_disease_swap
  + lambda_region_cf_contrastive * L_region_contrastive
  + lambda_dis * L_decorrelation
```

where:

- `L_region_adjustment` supervises the adjusted `p(m | do(z_d))`
- `L_region_context_stability` bounds unstable context sensitivity
- `L_region_disease_swap` supervises matched disease counterfactuals
- `L_region_contrastive` is a lesion-aware counterfactual contrastive loss

## Lesion-Aware Counterfactual Contrastive Learning

The current strongest model adds a lesion-aware counterfactual contrastive term:

- **anchor**: factual lesion-conditioned segmentation features
- **positive**: backdoor-adjusted same-disease features
- **hard negative**: matched disease-swapped counterfactual features

This term strengthens the causal mechanism in the segmentation feature space itself. It does not try to remove context; instead, it encourages disease-consistent lesion features to remain stable after adjustment, while separating them from disease-counterfactual lesion features.

## Current Mainline Interpretation

The current empirical evidence supports the following interpretation:

- `region_adjustment` is the strongest causal component
- `region_disease_swap` is the next most important causal term
- the lesion-aware contrastive term provides an additional gain, especially in HD95 and continuation-phase volume performance
- `region_cf_stability` behaves more like a geometric regularizer: it does not strongly improve Dice, but it helps stabilize HD95

This means the current framework should be described as a **causal region-adjusted segmentation model with matched disease counterfactual supervision and lesion-aware contrastive refinement**, rather than as a pure invariance model.
