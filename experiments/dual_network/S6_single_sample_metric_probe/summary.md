# S6_single_sample_metric_probe

## Purpose

Add minimal inversion diagnostics to the single-sample weak-form loop and run a longer probe with the current default weak-form configuration. The diagnostics are printed only and are not used in the training loss.

This experiment checks whether `mu_pred` moves closer to `mu_label`, or whether the weak-form losses can decrease while the material inversion remains poor.

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

## Configuration

```text
sample_index = 0
outer_steps = 30
phi_steps = 30
mu_steps = 30
test_radius = 5.0
center_mode = three
test centers = [-5, 5], [0, 5], [5, 5]
```

Command:

```powershell
python minimal_dual_single_sample_loop.py --npz-path experiments/dual_network/S3_single_sample_weak_form_probe/data/training_data_train.npz --sample-index 0 --outer-steps 30 --phi-steps 30 --mu-steps 30 --test-radius 5.0 --center-mode three
```

Full stdout:

```text
experiments/dual_network/S6_single_sample_metric_probe/run_default_metrics.txt
```

## Diagnostics

The following values are diagnostic only:

- `mu_mse = mean((mu_pred - mu_label)^2)`
- `mu_mae = mean(abs(mu_pred - mu_label))`
- `defect_area_pred = count(mu_pred < 500)`
- `defect_area_label = count(mu_label < 500)`
- `defect_iou` with threshold `500`

## Results

| metric | first outer | final outer |
| --- | ---: | ---: |
| loss_phi | 2.358378e+00 | 1.417249e-01 |
| loss_mu | 5.014572e-02 | 1.033289e-03 |
| mu_pred_min | 1.548524e+02 | 1.046731e+00 |
| mu_pred_max | 6.278730e+02 | 1.029179e+02 |
| mu_label_min | 1.000000e+00 | 1.000000e+00 |
| mu_label_max | 1.000000e+03 | 1.000000e+03 |
| mu_mse | 4.656037e+05 | 9.337209e+05 |
| mu_mae | 6.618752e+02 | 9.392122e+02 |
| defect_area_pred | 143 | 200 |
| defect_area_label | 11 | 11 |
| defect_iou | 1.986755e-02 | 5.500000e-02 |

All 30 outer steps were parsed successfully. No NaN/inf values were observed in the loss or diagnostic metrics.

## Initial Judgment

The weak-form and field losses are numerically stable and decrease over the run:

- `loss_phi` decreases by about one order of magnitude.
- `loss_mu` decreases from `5.014572e-02` to `1.033289e-03`.

However, the material diagnostics do not improve. `mu_mse` and `mu_mae` increase, and `defect_area_pred` grows to all 200 grid points while `defect_area_label` remains 11. The final `defect_iou` is only `5.500000e-02`.

This suggests the current weak-form-only `mu_step` has a visible numerical response, but it is not a useful defect-localized inversion response. In this probe, the loop mostly drives `mu_pred` toward a broad low-`mu` distribution while losses decrease.

## Current Limits

- Single sample only.
- Small `20 x 10` grid only.
- No saved model checkpoint.
- No image or field visualization.
- No quadrature weights.
- No `mu_label` prior in the training loss.
- Current `signals` still come from the analytic generator, not a strict PDE/finite-element forward solve.

## Next Step

If this branch continues, the next probe should add a lightweight `mu` prior or adjust the `mu_step` weighting while keeping the new diagnostics enabled. Another useful step is to save simple `mu_pred` and `mu_label` visualizations for manual inspection.
