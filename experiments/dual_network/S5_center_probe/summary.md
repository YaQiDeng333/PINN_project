# S5_center_probe

## Purpose

Compare how compact-support test center layout affects the single-sample weak-form closure probe. This experiment reuses the S3 generated `.npz` file, fixes `test_radius=5.0`, and changes only `--center-mode`.

This is a small numerical stability probe. It does not measure final inversion quality and does not compare against the main supervised baseline.

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
outer_steps = 10
phi_steps = 20
mu_steps = 20
test_radius = 5.0
```

Center modes:

- `three`: 3 centers, `[-5, 5]`, `[0, 5]`, `[5, 5]`
- `five`: 5 centers, `[-7.5, 5]`, `[-3.75, 5]`, `[0, 5]`, `[3.75, 5]`, `[7.5, 5]`
- `nine`: 9 centers on a `3 x 3` grid with `x = [-7.5, 0, 7.5]` and `y = [2.5, 5.0, 7.5]`

Run logs:

- `run_centers_three.txt`
- `run_centers_five.txt`
- `run_centers_nine.txt`

## Results

| center_mode | centers | loss_phi first | loss_phi last | loss_mu first | loss_mu last | mu_pred first min/max | mu_pred last min/max | mu_label min/max | finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| three | 3 | 2.373559e+00 | 2.129436e+00 | 3.367008e-01 | 1.166644e-02 | 2.007934e+02 / 4.908524e+02 | 6.485569e+00 / 5.621547e+02 | 1.000000e+00 / 1.000000e+03 | yes |
| five | 5 | 2.373559e+00 | 2.052791e+00 | 5.217735e-01 | 1.545732e-02 | 2.003384e+02 / 5.030875e+02 | 4.348617e+00 / 8.484542e+02 | 1.000000e+00 / 1.000000e+03 | yes |
| nine | 9 | 2.373559e+00 | 2.025729e+00 | 3.144610e-01 | 1.829428e-02 | 2.038694e+02 / 4.936988e+02 | 2.672376e+00 / 4.028873e+02 | 1.000000e+00 / 1.000000e+03 | yes |

All three runs completed normally and printed:

```text
Minimal dual-network single-sample loop passed.
```

No NaN/inf values were observed in the parsed loss or `mu_pred` ranges.

## Stability Note

All three center layouts are numerically stable in this small probe. The `three` layout gives the lowest final weak-form loss and keeps the final `mu_pred` range less extreme than `five`. The `nine` layout gives the lowest final `loss_phi`, but its final weak-form loss is the largest among the three and its `mu_pred_min` moves closer to the lower bound.

For the next probe, `three` is the most stable center mode by the current weak-form-loss criterion. This is not a final design choice for real training.

## Current Limits

- Single sample only.
- Small grid only.
- Fixed `test_radius=5.0`.
- No quadrature weights.
- No saved model checkpoint.
- No image or field visualization.
- No defect reconstruction metric.
