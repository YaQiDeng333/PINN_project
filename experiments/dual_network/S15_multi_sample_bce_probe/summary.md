# S15 Multi-Sample BCE Mask Prior Diagnostic

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 3 --val-samples 0 --test-samples 0 --grid-x 20 --grid-y 10 --output-dir experiments\dual_network\S15_multi_sample_bce_probe\data --seed 456
```

Generated files:

- `experiments/dual_network/S15_multi_sample_bce_probe/data/training_data_train.npz`
- `experiments/dual_network/S15_multi_sample_bce_probe/data/training_data_val.npz`
- `experiments/dual_network/S15_multi_sample_bce_probe/data/training_data_test.npz`

Train npz fields:

- `signals`: shape `[3, 20]`
- `mu_maps`: shape `[3, 10, 20]`
- `defect_types`: shape `[3]`
- `metadata`: shape `[3]`
- `x`: shape `[20]`
- `y`: shape `[10]`

## Fixed Loop Configuration

- `outer_steps=30`
- `phi_steps=30`
- `mu_steps=30`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`
- `mask_prior_temperature=50.0`

Compared modes:

- baseline: `lambda_mask_bce_prior=0.0`
- bce: `lambda_mask_bce_prior=1.0`

## Results

| sample | mode | label area | defect_area_pred | defect_iou | mu_mse | mu_mae | pred centroid | label centroid | NaN/inf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | baseline | 8 | 75 | 1.066667e-01 | 3.315026e+05 | 3.582430e+02 | (-2.368421, 6.622222) | (3.750000, 7.361111) | no |
| 0 | bce | 8 | 11 | 7.272727e-01 | 1.991005e+04 | 6.592333e+01 | (3.803828, 8.080808) | (3.750000, 7.361111) | no |
| 1 | baseline | 5 | 54 | 9.259259e-02 | 2.416857e+05 | 2.955117e+02 | (0.409357, 6.790123) | (-3.000000, 6.888888) | no |
| 1 | bce | 5 | 8 | 6.250000e-01 | 2.236068e+04 | 8.266766e+01 | (-3.552632, 7.916667) | (-3.000000, 6.888888) | no |
| 2 | baseline | 5 | 63 | 7.936508e-02 | 2.870101e+05 | 3.136474e+02 | (-1.090226, 6.613757) | (-0.789474, 3.777778) | no |
| 2 | bce | 5 | 5 | 1.000000e+00 | 5.665255e+03 | 3.456652e+01 | (-0.789474, 3.777778) | (-0.789474, 3.777778) | no |

## Interpretation

- BCE improves all three samples relative to baseline.
- `defect_area_pred` drops from 75/54/63 to 11/8/5, much closer to label areas 8/5/5.
- IoU improves from 0.1067/0.0926/0.0794 to 0.7273/0.6250/1.0000.
- `mu_mse` and `mu_mae` drop by roughly one order of magnitude in all three samples.
- Predicted centroids move much closer to the label centroids under BCE, especially sample 2.

## Current Judgment

The BCE mask prior is a stable semi-supervised upper-bound signal across this 3-sample probe. It consistently suppresses false positives, improves IoU, and reduces `mu_mse/mu_mae`.

This does not validate the unsupervised weak-form branch. The BCE term uses `mu_label < 500`, so it should be treated as a diagnostic or semi-supervised ceiling. The result mainly shows that the current weak-form + MuNet update lacks enough label-free false-positive suppression and localization pressure.

## Next Step

Stop adding stronger supervised priors in this prototype. The useful conclusion is now clear enough: BCE-style local mask information can correct the failure mode, while weak-form plus area/Dice priors alone remains too diffuse. The next work should be a stage summary or redesign of label-free localization/regularization, not another prior sweep.
