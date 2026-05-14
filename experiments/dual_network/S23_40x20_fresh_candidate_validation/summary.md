# S23 Fresh 40x20 Candidate Validation

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 20 --val-samples 0 --test-samples 0 --grid-x 40 --grid-y 20 --output-dir experiments\dual_network\S23_40x20_fresh_candidate_validation\data --seed 1023
```

Generated train data:

- `experiments/dual_network/S23_40x20_fresh_candidate_validation/data/training_data_train.npz`

## Candidate Configurations

Both runs used:

- `sample_indices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- `outer_steps=30`
- `phi_steps=30`
- `mu_steps=30`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`

Compared candidates:

- `temp25_lambda1`: `lambda_mask_bce_prior=1.0`, `mask_prior_temperature=25.0`
- `temp50_lambda3`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=50.0`

## Twenty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| temp25_lambda1 | 9.560439e-01 | 3.555000e+01 | 3.528870e+04 | 1.481365e+02 |
| temp50_lambda3 | 9.283914e-01 | 3.530000e+01 | 1.459990e+04 | 7.058641e+01 |

## Per-Sample IoU

| sample | temp25_lambda1 | temp50_lambda3 |
| ---: | ---: | ---: |
| 0 | 1.000000e+00 | 1.000000e+00 |
| 1 | 9.428570e-01 | 9.428570e-01 |
| 2 | 9.375000e-01 | 9.062500e-01 |
| 3 | 9.600000e-01 | 9.615380e-01 |
| 4 | 8.695650e-01 | 8.043480e-01 |
| 5 | 1.000000e+00 | 1.000000e+00 |
| 6 | 7.368420e-01 | 6.750000e-01 |
| 7 | 1.000000e+00 | 1.000000e+00 |
| 8 | 9.811320e-01 | 9.811320e-01 |
| 9 | 1.000000e+00 | 1.000000e+00 |
| 10 | 1.000000e+00 | 8.636360e-01 |
| 11 | 1.000000e+00 | 9.761900e-01 |
| 12 | 1.000000e+00 | 8.461540e-01 |
| 13 | 9.859150e-01 | 9.859150e-01 |
| 14 | 1.000000e+00 | 1.000000e+00 |
| 15 | 9.375000e-01 | 9.361700e-01 |
| 16 | 9.000000e-01 | 9.000000e-01 |
| 17 | 1.000000e+00 | 9.666670e-01 |
| 18 | 8.695650e-01 | 8.636360e-01 |
| 19 | 1.000000e+00 | 9.583330e-01 |

## Stability Check

- Both candidate runs completed successfully.
- No NaN/inf was observed in the aggregated metrics.
- `metrics.csv` was generated for both runs.
- No model checkpoints or weights were generated.
- No obvious failure sample was observed. The weakest sample is sample 6, with IoU `7.368420e-01` for `temp25_lambda1` and `6.750000e-01` for `temp50_lambda3`.

## Current Judgment

- `temp25_lambda1` has better average IoU and is the IoU-oriented default candidate for 40x20.
- `temp50_lambda3` has much lower `mu_mse` and `mu_mae`, so it remains the continuous-`mu` error-oriented candidate.
- S23 reproduces the S21 pattern on a fresh 40x20 / 20-sample dataset: sharper mask temperature is better for IoU, while stronger BCE weight is better for continuous material error.
- These results are still semi-supervised / diagnostic upper-bound results because BCE and mask priors use `mu_label < 500`. They are not proof of unsupervised weak-form inversion success.
