# S18 Runner 20-Sample BCE Diagnostic

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 20 --val-samples 0 --test-samples 0 --grid-x 20 --grid-y 10 --output-dir experiments\dual_network\S18_runner_20sample_bce_probe\data --seed 1018
```

Generated train data:

- `experiments/dual_network/S18_runner_20sample_bce_probe/data/training_data_train.npz`

## Runner Configuration

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
- `mask_prior_temperature=50.0`

Compared runs:

- baseline: `lambda_mask_bce_prior=0.0`
- bce: `lambda_mask_bce_prior=1.0`

## Twenty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.226982e-01 | 6.460000e+01 | 2.826850e+05 | 3.129512e+02 |
| bce | 8.934921e-01 | 9.100000e+00 | 1.558897e+04 | 6.164804e+01 |

## Per-Sample IoU and Area

| sample | baseline IoU | bce IoU | baseline area | bce area | label area |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.052632e-01 | 1.000000e+00 | 57 | 6 | 6 |
| 1 | 2.553191e-01 | 6.000000e-01 | 47 | 20 | 12 |
| 2 | 1.250000e-01 | 8.888889e-01 | 64 | 9 | 8 |
| 3 | 1.071429e-01 | 5.555556e-01 | 56 | 8 | 6 |
| 4 | 1.095890e-01 | 8.000000e-01 | 73 | 10 | 8 |
| 5 | 9.230769e-02 | 1.000000e+00 | 65 | 6 | 6 |
| 6 | 8.571429e-02 | 1.000000e+00 | 103 | 11 | 11 |
| 7 | 1.159420e-01 | 5.333334e-01 | 69 | 15 | 8 |
| 8 | 6.153846e-02 | 1.000000e+00 | 65 | 4 | 4 |
| 9 | 2.153846e-01 | 1.000000e+00 | 65 | 14 | 14 |
| 10 | 1.562500e-01 | 1.000000e+00 | 64 | 10 | 10 |
| 11 | 2.592593e-01 | 7.777778e-01 | 54 | 18 | 14 |
| 12 | 6.896552e-02 | 1.000000e+00 | 58 | 4 | 4 |
| 13 | 8.620690e-02 | 7.142857e-01 | 58 | 7 | 5 |
| 14 | 1.000000e-01 | 1.000000e+00 | 60 | 6 | 6 |
| 15 | 1.000000e-01 | 1.000000e+00 | 60 | 6 | 6 |
| 16 | 6.849315e-02 | 1.000000e+00 | 73 | 5 | 5 |
| 17 | 2.343750e-01 | 1.000000e+00 | 64 | 15 | 15 |
| 18 | 3.703704e-02 | 1.000000e+00 | 80 | 4 | 4 |
| 19 | 7.017544e-02 | 1.000000e+00 | 57 | 4 | 4 |

## Per-Sample Error

| sample | baseline mu_mse | bce mu_mse | baseline mu_mae | bce mu_mae |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.482180e+05 | 5.216209e+03 | 2.641570e+02 | 4.009204e+01 |
| 1 | 1.738169e+05 | 4.820249e+04 | 1.865176e+02 | 9.943353e+01 |
| 2 | 2.769270e+05 | 1.443606e+04 | 3.008988e+02 | 6.519614e+01 |
| 3 | 2.358991e+05 | 2.464177e+04 | 2.656204e+02 | 9.591169e+01 |
| 4 | 3.274067e+05 | 2.298703e+04 | 3.750750e+02 | 9.241503e+01 |
| 5 | 2.916939e+05 | 1.357173e+04 | 3.156667e+02 | 5.005817e+01 |
| 6 | 4.726067e+05 | 1.305889e+04 | 5.371161e+02 | 6.824725e+01 |
| 7 | 3.070219e+05 | 3.699384e+04 | 3.446559e+02 | 9.601849e+01 |
| 8 | 3.021375e+05 | 4.354118e+03 | 3.244084e+02 | 3.435485e+01 |
| 9 | 2.553408e+05 | 9.040287e+03 | 2.894380e+02 | 5.594112e+01 |
| 10 | 2.741114e+05 | 1.347540e+04 | 3.117853e+02 | 6.732489e+01 |
| 11 | 2.009259e+05 | 3.092584e+04 | 2.360518e+02 | 7.956229e+01 |
| 12 | 2.684114e+05 | 4.023753e+03 | 2.815358e+02 | 3.099686e+01 |
| 13 | 2.648718e+05 | 1.983625e+04 | 2.921389e+02 | 7.531606e+01 |
| 14 | 2.619539e+05 | 4.928463e+03 | 3.013324e+02 | 4.296000e+01 |
| 15 | 2.642105e+05 | 1.520206e+04 | 2.772796e+02 | 7.884819e+01 |
| 16 | 3.402730e+05 | 8.306302e+03 | 3.796808e+02 | 3.517459e+01 |
| 17 | 2.490810e+05 | 1.039347e+04 | 2.894496e+02 | 5.960238e+01 |
| 18 | 3.818876e+05 | 8.324033e+03 | 4.108029e+02 | 3.492486e+01 |
| 19 | 2.569042e+05 | 3.861469e+03 | 2.754127e+02 | 3.058241e+01 |

## Interpretation

BCE is better than baseline on all 20 samples for IoU, area control, `mu_mse`, and `mu_mae`.

The weakest BCE IoU cases are:

- sample 7: IoU improves from 1.159420e-01 to 5.333334e-01, but predicted area is 15 vs label area 8;
- sample 3: IoU improves from 1.071429e-01 to 5.555556e-01;
- sample 1: IoU improves from 2.553191e-01 to 6.000000e-01, but predicted area is 20 vs label area 12.

These are not regressions, but they remain useful failure samples for later inspection.

## Current Conclusion

The S17 conclusion holds on 20 samples: BCE mask prior is a stable semi-supervised diagnostic upper-bound signal. It strongly suppresses false positives and improves localization/error metrics.

This remains a semi-supervised result. BCE uses `mu_label < 500`, so this experiment does not prove unsupervised weak-form inversion success. It shows that the current branch needs a label-free equivalent of local mask supervision or false-positive suppression if the target is an unsupervised method.

