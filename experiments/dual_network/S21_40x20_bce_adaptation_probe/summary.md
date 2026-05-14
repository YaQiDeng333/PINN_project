# S21 40x20 BCE Adaptation Probe

## Data Source

This experiment reuses the S20 40x20 train dataset:

- `experiments/dual_network/S20_runner_20sample_40x20_bce_probe/data/training_data_train.npz`

No new data was generated for S21.

## S20 BCE Reference

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| S20 bce | 1.739786e-01 | 2.125500e+02 | 2.053370e+05 | 3.794331e+02 |

S20 used `outer_steps=20`, `phi_steps=20`, `mu_steps=20`, `mask_prior_temperature=50.0`, and `lambda_mask_bce_prior=1.0`.

## Runner Configuration

All S21 runs use:

- `sample_indices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`
- `outer_steps=30`
- `phi_steps=30`
- `mu_steps=30`

Compared runs:

- `bce_30steps_temp50`: `lambda_mask_bce_prior=1.0`, `mask_prior_temperature=50.0`
- `bce_30steps_temp25`: `lambda_mask_bce_prior=1.0`, `mask_prior_temperature=25.0`
- `bce_30steps_lambda3`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=50.0`

## Twenty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| S20 bce reference | 1.739786e-01 | 2.125500e+02 | 2.053370e+05 | 3.794331e+02 |
| bce_30steps_temp50 | 5.440697e-01 | 6.380000e+01 | 5.094976e+04 | 1.127006e+02 |
| bce_30steps_temp25 | 9.262554e-01 | 3.290000e+01 | 3.503892e+04 | 1.461498e+02 |
| bce_30steps_lambda3 | 9.053153e-01 | 3.280000e+01 | 1.327051e+04 | 6.555141e+01 |

## Per-Sample IoU

| sample | bce_30steps_temp50 | bce_30steps_temp25 | bce_30steps_lambda3 |
| ---: | ---: | ---: | ---: |
| 0 | 3.333333e-01 | 7.826087e-01 | 7.083333e-01 |
| 1 | 6.333333e-01 | 9.500000e-01 | 9.500000e-01 |
| 2 | 5.135135e-01 | 8.000000e-01 | 7.916667e-01 |
| 3 | 3.773585e-01 | 6.176471e-01 | 5.882353e-01 |
| 4 | 6.875000e-01 | 9.230769e-01 | 9.230769e-01 |
| 5 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 |
| 6 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 |
| 7 | 5.192308e-01 | 1.000000e+00 | 1.000000e+00 |
| 8 | 9.322034e-01 | 1.000000e+00 | 9.824561e-01 |
| 9 | 1.481481e-01 | 1.000000e+00 | 9.500000e-01 |
| 10 | 4.186047e-01 | 8.260870e-01 | 7.826087e-01 |
| 11 | 4.473684e-01 | 8.947368e-01 | 8.947368e-01 |
| 12 | 1.857143e-01 | 9.666667e-01 | 9.666667e-01 |
| 13 | 3.523810e-01 | 1.000000e+00 | 9.736842e-01 |
| 14 | 9.583333e-01 | 1.000000e+00 | 9.583333e-01 |
| 15 | 7.500000e-01 | 8.500000e-01 | 8.000000e-01 |
| 16 | 4.320988e-01 | 1.000000e+00 | 9.500000e-01 |
| 17 | 3.020833e-01 | 9.142857e-01 | 9.142857e-01 |
| 18 | 5.600000e-01 | 1.000000e+00 | 1.000000e+00 |
| 19 | 3.301887e-01 | 1.000000e+00 | 9.722222e-01 |

## Stability Check

- All three S21 runs completed successfully.
- No NaN/inf was observed in the aggregated metrics.
- `metrics.csv` was generated for all three runs.
- No model checkpoints or weights were generated.

## Current Judgment

- Increasing from `20/20/20` to `30/30/30` steps substantially improves the 40x20 BCE result: IoU rises from the S20 reference `1.739786e-01` to `5.440697e-01`.
- Lowering `mask_prior_temperature` to `25.0` gives the best average IoU at `9.262554e-01` and strongly improves predicted area control.
- Increasing `lambda_mask_bce_prior` to `3.0` gives the best `mu_mse` and `mu_mae`, while average IoU remains high at `9.053153e-01`.
- The largest IoU improvement comes from `bce_30steps_temp25`.
- The 40x20 issue in S20 was strongly affected by resolution adaptation, especially training steps and mask-prior sharpness.
- These results still use `mu_label < 500` through BCE / mask priors, so they remain semi-supervised / diagnostic upper-bound results, not proof of unsupervised weak-form inversion success.
