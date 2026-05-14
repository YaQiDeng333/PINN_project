# S4_radius_probe

## Purpose

Compare how compact-support test function radius affects the single-sample weak-form closure probe. This experiment reuses the S3 generated `.npz` file and changes only `--test-radius`.

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
test centers = [-5, 5], [0, 5], [5, 5]
```

Run logs:

- `run_radius_3.txt`
- `run_radius_5.txt`
- `run_radius_8.txt`

## Results

| test_radius | loss_phi first | loss_phi last | loss_mu first | loss_mu last | mu_pred first min/max | mu_pred last min/max | mu_label min/max | finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3.0 | 2.373559e+00 | 2.180017e+00 | 6.613781e-02 | 7.003976e-03 | 1.922332e+02 / 5.173950e+02 | 6.262142e+00 / 6.303340e+02 | 1.000000e+00 / 1.000000e+03 | yes |
| 5.0 | 2.373559e+00 | 2.129436e+00 | 3.367008e-01 | 1.166644e-02 | 2.007934e+02 / 4.908524e+02 | 6.485569e+00 / 5.621547e+02 | 1.000000e+00 / 1.000000e+03 | yes |
| 8.0 | 2.373559e+00 | 1.830089e+00 | 1.636217e-01 | 1.315128e-02 | 2.146140e+02 / 5.098311e+02 | 1.759705e+00 / 6.892171e+02 | 1.000000e+00 / 1.000000e+03 | yes |

All three runs completed normally and printed:

```text
Minimal dual-network single-sample loop passed.
```

No NaN/inf values were observed in the parsed loss or `mu_pred` ranges.

## Stability Note

All three radii are numerically stable in this small probe. Radius `3.0` gives the lowest final weak-form loss, while radius `8.0` gives the lowest final `loss_phi` but drives `mu_pred` to a more extreme range. Radius `5.0` is the most balanced setting in this run because it keeps finite losses, improves both losses, and produces a less extreme final `mu_pred` range than radius `8.0`.

This conclusion is only for one generated sample on a `20 x 10` grid. It is not a final choice for real training.

## Current Limits

- Single sample only.
- Small grid only.
- Fixed three test centers.
- No quadrature weights.
- No saved model checkpoint.
- No image or field visualization.
- No defect reconstruction metric.
