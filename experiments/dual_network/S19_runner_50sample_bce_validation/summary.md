# S19 Runner 50-Sample BCE Validation

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 50 --val-samples 0 --test-samples 0 --grid-x 20 --grid-y 10 --output-dir experiments\dual_network\S19_runner_50sample_bce_validation\data --seed 1019
```

Generated train data:

- `experiments/dual_network/S19_runner_50sample_bce_validation/data/training_data_train.npz`

## Runner Configuration

Both runs used:

- `sample_indices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49`
- `outer_steps=30`
- `phi_steps=30`
- `mu_steps=30`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`
- `mask_prior_temperature=50.0`

Compared runs:

- baseline: `lambda_mask_bce_prior=0.0`
- bce: `lambda_mask_bce_prior=1.0`

## Fifty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.101394e-01 | 6.504000e+01 | 2.871279e+05 | 3.216540e+02 |
| bce | 8.399348e-01 | 9.120000e+00 | 1.935842e+04 | 6.434378e+01 |

## Per-Sample IoU and Area

| sample | baseline IoU | bce IoU | baseline area | bce area | label area |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.860465e-01 | 1.000000e+00 | 86 | 16 | 16 |
| 1 | 1.451613e-01 | 8.181818e-01 | 62 | 11 | 9 |
| 2 | 8.219178e-02 | 1.000000e+00 | 73 | 6 | 6 |
| 3 | 6.153846e-02 | 1.000000e+00 | 65 | 4 | 4 |
| 4 | 6.250000e-02 | 1.000000e+00 | 79 | 6 | 6 |
| 5 | 5.714286e-02 | 1.000000e+00 | 70 | 4 | 4 |
| 6 | 1.538462e-01 | 7.692308e-01 | 65 | 13 | 10 |
| 7 | 1.363636e-01 | 1.000000e+00 | 66 | 9 | 9 |
| 8 | 6.896552e-02 | 1.000000e+00 | 87 | 6 | 6 |
| 9 | 1.694915e-01 | 1.000000e+00 | 59 | 10 | 10 |
| 10 | 5.172414e-02 | 6.000000e-01 | 58 | 5 | 3 |
| 11 | 1.764706e-01 | 1.000000e+00 | 51 | 9 | 9 |
| 12 | 1.666667e-01 | 4.736842e-01 | 54 | 19 | 9 |
| 13 | 8.620690e-02 | 8.000000e-01 | 58 | 4 | 5 |
| 14 | 1.818182e-01 | 5.000000e-01 | 44 | 13 | 8 |
| 15 | 1.666667e-01 | 1.000000e+00 | 54 | 9 | 9 |
| 16 | 5.970149e-02 | 8.000000e-01 | 67 | 5 | 4 |
| 17 | 1.562500e-01 | 1.000000e+00 | 64 | 10 | 10 |
| 18 | 1.129032e-01 | 1.000000e+00 | 62 | 7 | 7 |
| 19 | 1.875000e-01 | 8.000000e-01 | 64 | 15 | 12 |
| 20 | 7.462686e-02 | 4.285714e-01 | 67 | 5 | 5 |
| 21 | 1.111111e-01 | 1.000000e+00 | 81 | 9 | 9 |
| 22 | 1.343284e-01 | 6.000000e-01 | 67 | 15 | 9 |
| 23 | 1.967213e-01 | 1.000000e+00 | 61 | 12 | 12 |
| 24 | 4.166667e-02 | 4.285714e-01 | 72 | 7 | 3 |
| 25 | 2.962963e-01 | 5.714286e-01 | 54 | 28 | 16 |
| 26 | 1.139240e-01 | 8.181818e-01 | 79 | 11 | 9 |
| 27 | 3.333334e-02 | 1.000000e+00 | 60 | 2 | 2 |
| 28 | 5.172414e-02 | 5.000000e-01 | 58 | 6 | 3 |
| 29 | 8.955224e-02 | 1.000000e+00 | 67 | 6 | 6 |
| 30 | 8.474576e-02 | 8.333333e-01 | 59 | 6 | 5 |
| 31 | 7.500000e-02 | 1.000000e+00 | 80 | 6 | 6 |
| 32 | 1.250000e-01 | 1.000000e+00 | 64 | 8 | 8 |
| 33 | 9.090909e-02 | 1.000000e+00 | 66 | 6 | 6 |
| 34 | 7.575758e-02 | 1.851852e-01 | 66 | 27 | 5 |
| 35 | 1.066667e-01 | 1.000000e+00 | 75 | 8 | 8 |
| 36 | 1.034483e-01 | 4.000000e-01 | 58 | 15 | 6 |
| 37 | 1.617647e-01 | 8.333333e-01 | 68 | 11 | 11 |
| 38 | 4.109589e-02 | 1.000000e+00 | 73 | 3 | 3 |
| 39 | 1.071429e-01 | 1.000000e+00 | 56 | 6 | 6 |
| 40 | 9.859155e-02 | 7.777778e-01 | 71 | 9 | 7 |
| 41 | 7.142857e-02 | 8.000000e-01 | 56 | 5 | 4 |
| 42 | 4.411765e-02 | 1.000000e+00 | 68 | 3 | 3 |
| 43 | 2.195122e-01 | 1.000000e+00 | 41 | 9 | 9 |
| 44 | 6.250000e-02 | 1.000000e+00 | 79 | 6 | 6 |
| 45 | 1.166667e-01 | 2.592593e-01 | 60 | 27 | 7 |
| 46 | 8.108108e-02 | 1.000000e+00 | 74 | 6 | 6 |
| 47 | 8.000000e-02 | 1.000000e+00 | 50 | 4 | 4 |
| 48 | 4.705882e-02 | 1.000000e+00 | 85 | 4 | 4 |
| 49 | 1.020408e-01 | 1.000000e+00 | 49 | 5 | 5 |

## Stability Check

- BCE improves IoU over baseline on all 50 samples.
- BCE lowers `mu_mse` and `mu_mae` over baseline on all 50 samples.
- BCE reduces average predicted defect area from `65.04` to `9.12`.
- No NaN/inf was observed in metrics or run logs.
- `metrics.csv` was generated for both baseline and BCE runs.

## Failure Samples

BCE remains clearly better than baseline, but several samples still have lower final IoU:

- sample 34: BCE IoU `1.851852e-01`, predicted area 27 vs label area 5;
- sample 45: BCE IoU `2.592593e-01`, predicted area 27 vs label area 7;
- sample 36: BCE IoU `4.000000e-01`, predicted area 15 vs label area 6;
- sample 20 and sample 24: BCE IoU `4.285714e-01`;
- sample 12: BCE IoU `4.736842e-01`, predicted area 19 vs label area 9.

These are not regressions relative to baseline, but they are useful failure cases for later inspection.

## Current Conclusion

The S18 conclusion holds at 50 samples: BCE mask prior provides a stable semi-supervised diagnostic upper-bound signal. It strongly improves IoU, suppresses false positives, reduces predicted defect area, and lowers `mu_mse/mu_mae`.

This remains a semi-supervised result. BCE uses `mu_label < 500`, so this experiment does not prove unsupervised weak-form inversion success. It shows that the current branch benefits from a local mask-like signal, and that a label-free equivalent of this false-positive suppression remains the key open problem for an unsupervised variant.
