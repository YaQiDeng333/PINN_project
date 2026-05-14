# S27 80x40 Fresh Candidate Validation

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 20 --val-samples 0 --test-samples 0 --grid-x 80 --grid-y 40 --output-dir experiments\dual_network\S27_80x40_fresh_candidate_validation\data --seed 1027
```

Generated train data:

- `experiments/dual_network/S27_80x40_fresh_candidate_validation/data/training_data_train.npz`

The generated train file contains 20 samples with `grid_x=80` and `grid_y=40`.

## Runner Configuration

All runs used:

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

- `baseline`: `lambda_mask_bce_prior=0.0`, `mask_prior_temperature=50.0`
- `temp25_lambda3`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=25.0`
- `temp20_lambda3`: `lambda_mask_bce_prior=3.0`, `mask_prior_temperature=20.0`

## Twenty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.115430e-01 | 1.115400e+03 | 3.065218e+05 | 3.569112e+02 |
| temp25_lambda3 | 8.656310e-01 | 1.327500e+02 | 4.774737e+04 | 1.879750e+02 |
| temp20_lambda3 | 8.693352e-01 | 1.322500e+02 | 6.314803e+04 | 2.269721e+02 |

## Per-Sample IoU

| sample | baseline | temp25_lambda3 | temp20_lambda3 |
| ---: | ---: | ---: | ---: |
| 0 | 1.018520e-01 | 9.112900e-01 | 9.193550e-01 |
| 1 | 1.424550e-01 | 1.000000e+00 | 1.000000e+00 |
| 2 | 4.163400e-02 | 7.076920e-01 | 7.076920e-01 |
| 3 | 1.335080e-01 | 9.537040e-01 | 9.528300e-01 |
| 4 | 1.885130e-01 | 9.818180e-01 | 9.854550e-01 |
| 5 | 1.010230e-01 | 7.346940e-01 | 7.373740e-01 |
| 6 | 1.168160e-01 | 8.383840e-01 | 8.217820e-01 |
| 7 | 8.070400e-02 | 7.555560e-01 | 7.500000e-01 |
| 8 | 1.786410e-01 | 9.704430e-01 | 9.800990e-01 |
| 9 | 1.297590e-01 | 8.722630e-01 | 8.750000e-01 |
| 10 | 1.712180e-01 | 9.642860e-01 | 9.642860e-01 |
| 11 | 6.513900e-02 | 8.474580e-01 | 8.771930e-01 |
| 12 | 8.755300e-02 | 9.655170e-01 | 9.655170e-01 |
| 13 | 1.312450e-01 | 9.764710e-01 | 9.940480e-01 |
| 14 | 9.765900e-02 | 7.272730e-01 | 7.297300e-01 |
| 15 | 7.120100e-02 | 7.111110e-01 | 7.111110e-01 |
| 16 | 5.824600e-02 | 8.153850e-01 | 7.969920e-01 |
| 17 | 9.461500e-02 | 6.792450e-01 | 7.077920e-01 |
| 18 | 1.144580e-01 | 9.895830e-01 | 1.000000e+00 |
| 19 | 1.246200e-01 | 9.104480e-01 | 9.104480e-01 |

## Stability Check

- All three runner jobs completed successfully.
- Each `metrics.csv` contains 20 rows.
- No NaN/inf was observed in aggregated metrics.
- No model checkpoints or weights were generated.
- Both BCE candidates improve over baseline on all 20 samples.
- Weak samples remain around the same cases for both candidates, especially samples 2, 5, 14, 15, and 17.

## Current Recommendation

- IoU priority: choose `temp20_lambda3`; it has the best average IoU at `8.693352e-01`.
- Continuous `mu` error priority: choose `temp25_lambda3`; it has lower `mu_mse` and `mu_mae`.
- Overall default candidate: choose `temp25_lambda3` for now, because its IoU is only slightly below `temp20_lambda3` while its continuous `mu` errors are substantially lower.
- S27 validates that S26's two candidate configurations are stable on a fresh 20-sample `80x40` dataset.
- The result remains a semi-supervised / diagnostic upper-bound result because BCE and mask priors use `mu_label < 500`; it does not prove unsupervised weak-form inversion success.
