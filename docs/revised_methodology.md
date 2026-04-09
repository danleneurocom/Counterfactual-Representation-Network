# Revised Methodology

## Causal Correction

The original proposal used the disease latent `z_d` as the only input to the prediction heads and treated the context latent `z_c` as information that should be removed from inference. The supervisor feedback points out the causal issue: in a causal graph, confounders are often also causes of the label. Removing them can remove genuine causal parents of `Y`.

The corrected framing is **causal effect estimation under latent confounding**, not confounder removal.

## Model

For an image `x`, the model learns:

```text
z_d = E_d(x)
z_c = E_c(x)
y_hat = f_cls(z_d, z_c)
m_hat = f_seg(z_d, z_c)
x_hat = G(z_d, z_c)
```

`z_d` is encouraged to represent disease-related factors. `z_c` represents context, acquisition, anatomy, scanner style, and other latent factors that may be associated with both the image and label.

## Latent Backdoor Adjustment

The target estimand is:

```text
p(y | do(z_d)) = integral p(y | z_d, z_c) p(z_c) dz_c
```

The minibatch approximation used in this scaffold is:

```text
p(y | do(z_d_i)) ~= (1 / K) sum_k p(y | z_d_i, z_c_k)
```

where `z_c_k` is drawn from the current minibatch. This keeps `z_c` in the prediction model, but estimates the disease-factor intervention by averaging over context.

## Counterfactual Swapping

Context swaps should not enforce strict invariance:

```text
f(z_d_i, z_c_i) == f(z_d_i, z_c_j)
```

That assumption implies zero causal influence from `z_c`. Instead, this implementation uses a bounded stability penalty:

```text
max(abs(p_i - p_ij) - margin, 0)
```

Small context effects are allowed. Large unstable context effects are penalized.

## Training Objective

The implemented objective combines:

```text
L_total =
    lambda_cls * BCE(f_cls(z_d, z_c), y)
  + lambda_adjustment * BCE(E_zc[f_cls(z_d, z_c)], y)
  + lambda_cf_stability * bounded_context_swap(f_cls)
  + lambda_disease_swap * BCE(f_cls(z_d_donor, z_c), y_donor)
  + lambda_seg * segmentation_loss(f_seg(z_d, z_c), m)
  + lambda_rec * reconstruction_loss(G(z_d, z_c), x)
  + lambda_dis * decorrelation(z_d, z_c)
```

The adjustment and bounded stability terms are the key methodological fixes.

