# S22 40x20 BCE Combo Adaptation Probe

## Data Source

This experiment reuses the S20 40x20 train dataset:

- `experiments/dual_network/S20_runner_20sample_40x20_bce_probe/data/training_data_train.npz`

No new data was generated for S22.

## S21 References

| reference | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| S21 bce_30steps_temp25 | 9.262554e-01 | 3.290000e+01 | 3.503892e+04 | 1.461498e+02 |
| S21 bce_30steps_lambda3 | 9.053153e-01 | 3.280000e+01 | 1.327051e+04 | 6.555141e+01 |

S21 showed that `mask_prior_temperature=25.0` gave the best IoU, while `lambda_mask_bce_prior=3.0` gave the best `mu_mse/mu_mae`.

## Runner Configuration

All S22 runs use:

- `sample_indices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- `outer_steps=30`
- `phi_steps=30`
- `mu_steps=30`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`

Compared runs:

- `combo_temp25_lambda3`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=25.0`
- `combo_temp20_lambda3`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=20.0`
- `combo_temp25_lambda5`: `lambda_mask_bce_prior=5.0`, `mask_prior_temperature=25.0`

## Twenty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| combo_temp25_lambda3 | 9.177421e-01 | 3.260000e+01 | 3.814877e+04 | 1.629943e+02 |
| combo_temp20_lambda3 | 9.172025e-01 | 3.265000e+01 | 5.109617e+04 | 1.986578e+02 |
| combo_temp25_lambda5 | 9.139707e-01 | 3.230000e+01 | 4.181617e+04 | 1.753928e+02 |

## Per-Sample IoU

| sample | combo_temp25_lambda3 | combo_temp20_lambda3 | combo_temp25_lambda5 |
| ---: | ---: | ---: | ---: |
| 0 | 7.391304e-01 | 7.083333e-01 | 7.391304e-01 |
| 1 | 9.500000e-01 | 9.500000e-01 | 9.500000e-01 |
| 2 | 7.600000e-01 | 7.600000e-01 | 8.260870e-01 |
| 3 | 6.060606e-01 | 6.060606e-01 | 5.625000e-01 |
| 4 | 9.230769e-01 | 9.230769e-01 | 9.230769e-01 |
| 5 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 |
| 6 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 |
| 7 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 |
| 8 | 1.000000e+00 | 9.824561e-01 | 9.824561e-01 |
| 9 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 |
| 10 | 8.260870e-01 | 8.636360e-01 | 8.260870e-01 |
| 11 | 8.947370e-01 | 8.947370e-01 | 8.947370e-01 |
| 12 | 9.666670e-01 | 9.666670e-01 | 9.666670e-01 |
| 13 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 |
| 14 | 9.583330e-01 | 9.583330e-01 | 9.583330e-01 |
| 15 | 8.421050e-01 | 8.421050e-01 | 7.894740e-01 |
| 16 | 9.743590e-01 | 9.743590e-01 | 9.743590e-01 |
| 17 | 9.142860e-01 | 9.142860e-01 | 9.142860e-01 |
| 18 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 |
| 19 | 1.000000e+00 | 1.000000e+00 | 9.722220e-01 |

## Stability Check

- All three S22 runs completed successfully.
- No NaN/inf was observed in the aggregated metrics.
- `metrics.csv` was generated for all three runs.
- No model checkpoints or weights were generated.

## Current Judgment

- Among S22 combo runs, `combo_temp25_lambda3` is the best candidate: it has the highest average IoU and the lowest `mu_mse/mu_mae` among the three combinations.
- None of the S22 combinations strictly dominates the best S21 single-axis settings.
- Compared with S21 `bce_30steps_temp25`, combo settings slightly reduce IoU.
- Compared with S21 `bce_30steps_lambda3`, combo settings have worse `mu_mse/mu_mae`.
- If choosing only among S22 combo runs, use `combo_temp25_lambda3` as the next candidate.
- If choosing the best overall 40x20 setting so far, keep S21 `bce_30steps_temp25` for IoU-oriented experiments and S21 `bce_30steps_lambda3` for error-oriented experiments.
- These results remain semi-supervised / diagnostic upper-bound results because the BCE and mask priors use `mu_label < 500`.
