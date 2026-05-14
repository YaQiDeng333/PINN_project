# S8_stronger_area_prior_probe

## Purpose

Continue the S7 anti-collapse test with stronger `lambda_area_prior` values. The goal is to see whether a stronger area prior further suppresses whole-domain low-`mu` collapse and whether that also improves defect localization.

This is still a single-sample diagnostic branch experiment, not a final unsupervised inversion setting.

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
area_prior_temperature = 50.0
```

Area prior values:

- `lambda_area_prior = 1.0`
- `lambda_area_prior = 3.0`
- `lambda_area_prior = 10.0`

Run logs:

- `run_lambda_area_1.txt`
- `run_lambda_area_3.txt`
- `run_lambda_area_10.txt`

## Results

| lambda_area_prior | loss_phi first | loss_phi last | loss_mu first | loss_mu last | mu_mse first | mu_mse last | mu_mae first | mu_mae last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 2.358378e+00 | 1.106874e+00 | 2.704039e-01 | 1.202469e-01 | 2.044109e+05 | 3.731120e+05 | 4.486964e+02 | 4.114994e+02 |
| 3.0 | 2.358378e+00 | 1.505556e+00 | 3.397159e-01 | 1.261616e-01 | 1.694989e+05 | 2.928783e+05 | 4.046881e+02 | 3.140588e+02 |
| 10.0 | 2.358378e+00 | 1.649895e+00 | 4.018669e-01 | 2.390804e-01 | 1.559650e+05 | 2.515649e+05 | 3.847365e+02 | 2.794536e+02 |

| lambda_area_prior | defect_area_pred first | defect_area_pred last | defect_area_label | defect_iou first | defect_iou last | pred_fraction first | pred_fraction last | target_fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 20 | 77 | 11 | 0.000000e+00 | 7.317073e-02 | 2.464109e-01 | 3.862506e-01 | 5.500000e-02 |
| 3.0 | 0 | 50 | 11 | 0.000000e+00 | 0.000000e+00 | 1.448153e-01 | 2.499745e-01 | 5.500000e-02 |
| 10.0 | 0 | 41 | 11 | 0.000000e+00 | 0.000000e+00 | 1.227327e-01 | 2.050712e-01 | 5.500000e-02 |

All three runs completed normally and printed:

```text
Minimal dual-network single-sample loop passed.
```

No NaN/inf values were observed in the parsed loss or diagnostic metrics.

## Judgment

Stronger area prior continues to suppress whole-domain low-`mu` collapse:

- final `defect_area_pred` decreases from `77` at `lambda=1.0` to `50` at `lambda=3.0` and `41` at `lambda=10.0`;
- final `pred_defect_fraction` also moves closer to the target fraction `0.055`;
- final `mu_mse` and `mu_mae` improve as lambda increases.

However, `defect_iou` does not improve with stronger priors. It drops to `0.0` for `lambda=3.0` and `lambda=10.0`, meaning the predicted low-`mu` region is not overlapping the label defect under the threshold metric.

This suggests the stronger area prior can control area but does not solve localization. The main issue shifts from area collapse to defect placement.

## Candidate Setting

For the next step:

- `lambda_area_prior=1.0` remains the best balance if IoU is prioritized, because it is the only setting here with nonzero final IoU.
- `lambda_area_prior=10.0` gives the best area suppression and lowest `mu_mse` / `mu_mae`, but its IoU is zero, so it is not a good localization candidate without additional constraints.

## Current Limits

- The target area comes from `mu_label`, so this is a supervised diagnostic probe, not the final unsupervised formulation.
- Single sample only.
- Small `20 x 10` grid only.
- No saved model checkpoint.
- No visualization of predicted defect location.
- No quadrature weights.
- No comparison to the main supervised baseline.

## Next Step

If high lambda controls area but IoU stays low, the next step should be visualization or localization constraints rather than simply increasing `lambda_area_prior`. A useful S9 would save `mu_pred` / `mu_label` images for lambda `1.0`, `3.0`, and `10.0`.
