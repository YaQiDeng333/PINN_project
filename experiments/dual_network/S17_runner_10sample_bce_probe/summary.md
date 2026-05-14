# S17 Runner 10-Sample BCE Diagnostic

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 10 --val-samples 0 --test-samples 0 --grid-x 20 --grid-y 10 --output-dir experiments\dual_network\S17_runner_10sample_bce_probe\data --seed 789
```

Generated train data:

- `experiments/dual_network/S17_runner_10sample_bce_probe/data/training_data_train.npz`

## Runner Configuration

Both runs used:

- `sample_indices=0,1,2,3,4,5,6,7,8,9`
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

## Ten-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.145468e-01 | 6.280000e+01 | 2.766025e+05 | 3.173523e+02 |
| bce | 8.425641e-01 | 7.800000e+00 | 1.319349e+04 | 6.156954e+01 |

## Per-Sample IoU and Area

| sample | baseline IoU | bce IoU | baseline area | bce area | label area |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.547170e-02 | 1.000000e+00 | 53 | 4 | 4 |
| 1 | 8.219178e-02 | 8.333333e-01 | 73 | 5 | 6 |
| 2 | 5.714286e-02 | 8.000000e-01 | 70 | 5 | 4 |
| 3 | 1.250000e-01 | 1.000000e+00 | 64 | 8 | 8 |
| 4 | 9.589041e-02 | 1.000000e+00 | 73 | 7 | 7 |
| 5 | 2.027027e-01 | 1.000000e+00 | 74 | 15 | 15 |
| 6 | 1.698113e-01 | 6.923077e-01 | 53 | 13 | 9 |
| 7 | 1.063830e-01 | 3.000000e-01 | 47 | 8 | 5 |
| 8 | 8.333334e-02 | 8.000000e-01 | 60 | 4 | 5 |
| 9 | 1.475410e-01 | 1.000000e+00 | 61 | 9 | 9 |

## Per-Sample Error

| sample | baseline mu_mse | bce mu_mse | baseline mu_mae | bce mu_mae |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.422419e+05 | 4.084926e+03 | 2.875308e+02 | 4.631758e+01 |
| 1 | 3.416644e+05 | 1.211631e+04 | 4.134229e+02 | 5.098491e+01 |
| 2 | 3.256448e+05 | 1.174311e+04 | 3.436537e+02 | 5.076174e+01 |
| 3 | 2.781092e+05 | 5.956890e+03 | 3.086381e+02 | 4.739457e+01 |
| 4 | 3.229166e+05 | 5.314347e+03 | 3.635484e+02 | 4.717471e+01 |
| 5 | 2.985924e+05 | 1.232611e+04 | 3.459136e+02 | 6.942561e+01 |
| 6 | 2.211301e+05 | 2.620544e+04 | 2.601286e+02 | 7.957752e+01 |
| 7 | 2.065416e+05 | 3.453107e+04 | 2.481122e+02 | 1.165149e+02 |
| 8 | 2.658915e+05 | 1.017965e+04 | 2.887424e+02 | 5.419108e+01 |
| 9 | 2.632930e+05 | 9.477104e+03 | 3.138320e+02 | 5.335275e+01 |

## Interpretation

BCE is consistently better than baseline on all 10 samples for IoU, area control, `mu_mse`, and `mu_mae`.

The weakest BCE case is sample 7: IoU improves from 1.063830e-01 to 3.000000e-01, but remains substantially lower than the other BCE samples. Sample 6 is also less strong than the rest with IoU 6.923077e-01. These are not regressions, but they are useful failure cases for later inspection.

## Current Conclusion

The S15/S16 conclusion holds on 10 samples: BCE mask prior is a stable semi-supervised diagnostic upper-bound signal. It strongly suppresses false positives and improves localization/error metrics.

This is not proof of unsupervised weak-form inversion success. BCE uses `mu_label < 500`, so the result should be interpreted as evidence that the current branch needs a label-free equivalent of false-positive suppression or localization pressure.
