# S12_lightweight_mask_prior_probe

## Purpose

Test a lightweight mask prior as a diagnostic for the localization failure observed in S6-S11. The prior uses a soft Dice loss between a soft predicted defect mask and the label mask.

This is a supervised diagnostic experiment only. It is not the final unsupervised inversion formulation.

## Input Data

Reused `.npz`:

```text
experiments/dual_network/S3_single_sample_weak_form_probe/data/training_data_train.npz
```

## Shared Configuration

```text
sample_index = 0
outer_steps = 30
phi_steps = 30
mu_steps = 30
test_radius = 5.0
center_mode = three
lambda_area_prior = 1.0
area_prior_temperature = 50.0
mask_prior_temperature = 50.0
```

Mask prior:

```text
soft_defect = sigmoid((500.0 - mu) / mask_prior_temperature)
label_mask = (mu_label < 500).float()
dice_loss = 1 - (2 * sum(soft_defect * label_mask) + eps) / (sum(soft_defect) + sum(label_mask) + eps)
```

Tested values:

- `lambda_mask_prior = 0.0`
- `lambda_mask_prior = 0.01`
- `lambda_mask_prior = 0.1`
- `lambda_mask_prior = 1.0`

## Results

| lambda_mask_prior | final loss_phi | final loss_mu | final mu_mse | final mu_mae | defect_area_pred | defect_area_label | final defect_iou | dice_loss first | dice_loss last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1.106874e+00 | 1.202469e-01 | 3.731120e+05 | 4.114994e+02 | 77 | 11 | 7.317073e-02 | 9.658419e-01 | 8.652049e-01 |
| 0.01 | 1.244194e+00 | 9.823377e-02 | 3.427134e+05 | 3.657416e+02 | 66 | 11 | 4.054054e-02 | 9.649049e-01 | 9.220452e-01 |
| 0.1 | 1.193429e+00 | 2.020276e-01 | 3.566554e+05 | 3.904005e+02 | 75 | 11 | 7.500000e-02 | 9.543499e-01 | 8.605599e-01 |
| 1.0 | 1.438265e+00 | 8.082112e-01 | 2.803630e+05 | 3.275804e+02 | 66 | 11 | 1.666667e-01 | 8.014291e-01 | 7.238815e-01 |

All four runs completed normally. No NaN/inf values were observed in parsed metrics.

## Judgment

`lambda_mask_prior=1.0` is the most useful setting in this diagnostic:

- final IoU improves from `7.317073e-02` to `1.666667e-01`;
- final defect area decreases from `77` to `66`, still much larger than the label area `11`;
- final `mu_mse` and `mu_mae` improve compared with the no-mask-prior run;
- final Dice loss is the lowest of the four runs.

Smaller values are not consistently helpful. `lambda=0.01` reduces area and `mu_mae`, but IoU worsens. `lambda=0.1` is close to baseline.

## Current Conclusion

The mask prior clearly helps localization more than area prior alone, so the current branch is missing a local positioning constraint. However, even with `lambda_mask_prior=1.0`, the predicted defect area remains far larger than the label area. This means a localization prior helps, but the weak-form + mu-Net update still does not produce a compact defect.

If this direction continues, the next experiments should focus on weaker/localized priors, semi-supervised settings, or weak-form weights that encode local defect support more directly.
