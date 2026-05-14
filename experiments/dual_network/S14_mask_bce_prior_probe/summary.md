# S14 Mask BCE Prior Probe

## Data

- npz: `experiments/dual_network/S3_single_sample_weak_form_probe/data/training_data_train.npz`
- sample_index: 0
- center_mode: `three`
- test_radius: 5.0
- lambda_area_prior: 1.0
- lambda_mask_prior: 1.0
- area_prior_temperature: 50.0
- mask_prior_temperature: 50.0
- outer_steps: 30
- phi_steps: 30
- mu_steps: 30

## Results

| lambda_mask_bce_prior | final loss_phi | final loss_mu | final mu_mse | final mu_mae | defect_area_pred | defect_area_label | final defect_iou | pred centroid | label centroid | mask_bce_loss first -> final | NaN/inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 0.0 | 1.438265e+00 | 8.082112e-01 | 2.803630e+05 | 3.275804e+02 | 66 | 11 | 1.666667e-01 | (-1.698565, 6.313132) | (-7.248804, 4.040404) | 4.422840e-01 -> 2.657662e+00 | no |
| 0.1 | 1.533216e+00 | 9.675508e-01 | 2.331388e+05 | 2.859181e+02 | 57 | 11 | 1.929825e-01 | (-1.869806, 6.218324) | (-7.248804, 4.040404) | 3.613158e-01 -> 2.166750e+00 | no |
| 1.0 | 2.298001e+00 | 4.520448e-02 | 1.356608e+04 | 6.732818e+01 | 11 | 11 | 1.000000e+00 | (-7.248804, 4.040404) | (-7.248804, 4.040404) | 2.482112e-01 -> 4.156713e-03 | no |
| 3.0 | 2.297582e+00 | 5.923045e-02 | 1.495469e+04 | 7.611285e+01 | 11 | 11 | 1.000000e+00 | (-7.248804, 4.040404) | (-7.248804, 4.040404) | 2.311356e-01 -> 4.588332e-03 | no |

## Interpretation

- BCE mask prior strongly suppresses false positives once `lambda_mask_bce_prior >= 1.0`.
- `lambda_mask_bce_prior=1.0` gives the best numerical tradeoff in this single-sample diagnostic: `defect_area_pred` matches `defect_area_label`, IoU reaches 1.0, and `mu_mse/mu_mae` are the lowest among the four runs.
- `lambda_mask_bce_prior=3.0` also reaches IoU 1.0 and exact area, but has slightly worse `mu_mse/mu_mae` and a larger final `loss_mu` than 1.0.
- This result shows that the previous false-positive expansion can be controlled by a direct mask BCE prior. It also means the S14 improvement depends on supervised mask information and is not evidence that the weak-form branch alone solves localization.

## Current Judgment

- Most useful next candidate: `lambda_mask_bce_prior=1.0`.
- The main failure exposed by S6-S13 includes insufficient false-positive suppression. BCE fixes this in the oracle diagnostic setting.
- Since BCE uses `mu_label`, it should remain a diagnostic or semi-supervised option unless a label-free equivalent is designed.
