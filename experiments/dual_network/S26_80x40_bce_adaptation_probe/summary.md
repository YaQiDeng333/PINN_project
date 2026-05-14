# S26 80x40 BCE Adaptation Probe

## Data Source

S26 reuses the S25 `80x40` dataset:

- `experiments/dual_network/S25_80x40_10sample_feasibility_probe/data/training_data_train.npz`

The dataset contains 10 train samples with `grid_x=80` and `grid_y=40`.

## S25 Reference

S25 `temp25_lambda1` used `20/20/20` training steps and is the reference for this adaptation probe:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| S25 temp25_lambda1 | 5.102481e-01 | 2.611000e+02 | 1.519206e+05 | 3.708853e+02 |

## Runner Configuration

All S26 runs used:

- `sample_indices=0,1,2,3,4,5,6,7,8,9`
- `outer_steps=30`
- `phi_steps=30`
- `mu_steps=30`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`

Compared runs:

- `temp25_lambda1_30steps`: `lambda_mask_bce_prior=1.0`, `mask_prior_temperature=25.0`
- `temp25_lambda3_30steps`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=25.0`
- `temp20_lambda3_30steps`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=20.0`

## Ten-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| temp25_lambda1_30steps | 8.081555e-01 | 1.308000e+02 | 4.524894e+04 | 1.601105e+02 |
| temp25_lambda3_30steps | 8.706159e-01 | 1.068000e+02 | 4.723388e+04 | 1.832961e+02 |
| temp20_lambda3_30steps | 8.866546e-01 | 1.077000e+02 | 5.950246e+04 | 2.175080e+02 |

## Per-Sample IoU

| sample | temp25_lambda1_30steps | temp25_lambda3_30steps | temp20_lambda3_30steps |
| ---: | ---: | ---: | ---: |
| 0 | 8.220340e-01 | 8.220340e-01 | 8.362070e-01 |
| 1 | 7.835050e-01 | 7.524750e-01 | 7.572820e-01 |
| 2 | 8.431370e-01 | 8.431370e-01 | 8.627450e-01 |
| 3 | 9.333330e-01 | 9.180330e-01 | 9.180330e-01 |
| 4 | 8.765430e-01 | 8.641980e-01 | 8.809520e-01 |
| 5 | 8.823530e-01 | 8.823530e-01 | 8.823530e-01 |
| 6 | 5.463410e-01 | 8.838170e-01 | 9.739130e-01 |
| 7 | 8.969070e-01 | 9.062500e-01 | 8.969070e-01 |
| 8 | 9.822490e-01 | 9.822490e-01 | 9.880240e-01 |
| 9 | 5.151520e-01 | 8.516130e-01 | 8.701300e-01 |

## Stability Check

- All three runner jobs completed successfully.
- Each `metrics.csv` contains 10 rows.
- No NaN/inf was observed in aggregated metrics.
- No model checkpoints or weights were generated.

## Current Judgment

- All S26 configurations improve strongly over the S25 `20/20/20` reference, so S25 was substantially limited by training steps and high-resolution adaptation.
- `temp20_lambda3_30steps` gives the best average IoU: `8.866546e-01`.
- `temp25_lambda3_30steps` is the most balanced follow-up candidate: it is close to the best IoU, has the lowest average predicted defect area, and keeps lower `mu_mse/mu_mae` than `temp20_lambda3_30steps`.
- `temp25_lambda1_30steps` gives the lowest `mu_mse/mu_mae`, but its IoU remains weaker because samples 6 and 9 are still poor.
- The result remains a semi-supervised / diagnostic upper-bound result because BCE and mask priors use `mu_label < 500`; it does not prove unsupervised weak-form inversion success.
