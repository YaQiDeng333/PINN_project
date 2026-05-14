# S7_area_prior_probe

## Purpose

Test whether a lightweight soft area prior can reduce the whole-domain low-`mu` collapse observed in S6. The prior is added only to the `mu-step` and is not a full supervised `mu` loss.

The diagnostic metrics remain print-only, except for the area prior term when `lambda_area_prior > 0`.

## Input Data

Reused `.npz`:

```text
experiments/dual_network/S3_single_sample_weak_form_probe/data/training_data_train.npz
```

Fields:

```text
signals: (1, 20)
mu_maps: (1, 10, 20)
x: (20,)
y: (10,)
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

Area prior:

```text
soft_defect = sigmoid((500.0 - mu) / area_prior_temperature)
target_defect_fraction = mean(mu_label < 500)
pred_defect_fraction = mean(soft_defect)
area_prior_loss = (pred_defect_fraction - target_defect_fraction)^2
```

Run logs:

- `run_lambda_area_0.txt`
- `run_lambda_area_0p1.txt`
- `run_lambda_area_1.txt`

## Results

| lambda_area_prior | loss_phi first | loss_phi last | loss_mu first | loss_mu last | mu_mse first | mu_mse last | mu_mae first | mu_mae last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2.358378e+00 | 1.417249e-01 | 5.014572e-02 | 1.033289e-03 | 4.656037e+05 | 9.337209e+05 | 6.618752e+02 | 9.392122e+02 |
| 0.1 | 2.358378e+00 | 5.194988e-01 | 8.865374e-02 | 5.427388e-02 | 4.671314e+05 | 7.247131e+05 | 6.502485e+02 | 7.497108e+02 |
| 1.0 | 2.358378e+00 | 1.106874e+00 | 2.704039e-01 | 1.202469e-01 | 2.044109e+05 | 3.731120e+05 | 4.486964e+02 | 4.114994e+02 |

| lambda_area_prior | defect_area_pred first | defect_area_pred last | defect_area_label | defect_iou first | defect_iou last | pred_fraction first | pred_fraction last | target_fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 143 | 200 | 11 | 1.986755e-02 | 5.500000e-02 | 7.462106e-01 | 9.999436e-01 | 5.500000e-02 |
| 0.1 | 133 | 155 | 11 | 6.993007e-03 | 6.410257e-02 | 6.773760e-01 | 7.776257e-01 | 5.500000e-02 |
| 1.0 | 20 | 77 | 11 | 0.000000e+00 | 7.317073e-02 | 2.464109e-01 | 3.862506e-01 | 5.500000e-02 |

All three runs completed normally and printed:

```text
Minimal dual-network single-sample loop passed.
```

No NaN/inf values were observed in the parsed loss or diagnostic metrics.

## Initial Judgment

`lambda_area_prior=1.0` best suppresses the whole-domain low-`mu` collapse in this small probe:

- final `defect_area_pred` drops from `200` at `lambda=0.0` to `77`;
- final `pred_defect_fraction` drops from about `1.00` to about `0.386`;
- final `mu_mse` and `mu_mae` are substantially lower than the no-prior run;
- final `defect_iou` is still low, but it is the highest among the three runs.

The prior does not solve the inversion problem. It reduces collapse, but the predicted defect area is still much larger than the label area of `11`, and the IoU remains only `7.317073e-02`.

## Current Limits

- Single sample only.
- Small `20 x 10` grid only.
- The target defect fraction comes from `mu_label`; this is a diagnostic branch experiment, not the final unsupervised setting.
- No saved model checkpoint.
- No field visualization.
- No quadrature weights.
- No comparison to the main supervised baseline.

## Next Step

If this branch continues, keep `lambda_area_prior=1.0` as an experimental anti-collapse candidate and add visualization of `mu_pred` / `mu_label`. If IoU remains weak, consider a more local `mu` prior or redesign weak-form centers and quadrature weights.
