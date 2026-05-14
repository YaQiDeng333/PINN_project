# S13_label_centers_mask_prior_probe

## Purpose

Combine label-informed oracle centers with the lightweight mask prior to estimate an upper-bound diagnostic for the current single-sample weak-form framework.

S11 showed that label-informed centers improve centroid and IoU but leave the mask too large. S12 showed that the mask prior improves IoU and `mu_mse` / `mu_mae` but also leaves the area too large. S13 tests whether combining both signals gives a stronger result.

This is a diagnostic experiment only. It uses label information and is not the final unsupervised formulation.

## Input Data

Reused `.npz`:

```text
experiments/dual_network/S3_single_sample_weak_form_probe/data/training_data_train.npz
```

## Shared Configuration

```text
sample_index = 0
outer_steps = 30
phi_steps = 30
mu_steps = 30
test_radius = 5.0
lambda_area_prior = 1.0
area_prior_temperature = 50.0
mask_prior_temperature = 50.0
```

## Runs

| run | center_mode | lambda_mask_prior |
| --- | --- | ---: |
| `three_mask1` | `three` | 1.0 |
| `label_three_mask1` | `label_three` | 1.0 |
| `label_nine_mask1` | `label_nine` | 1.0 |
| `label_three_mask3` | `label_three` | 3.0 |

## Results

| run | final loss_phi | final loss_mu | final mu_mse | final mu_mae | defect_area_pred | defect_area_label | final defect_iou | pred centroid | label centroid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `three_mask1` | 1.438265e+00 | 8.082112e-01 | 2.803630e+05 | 3.275804e+02 | 66 | 11 | 1.666667e-01 | `(-1.698565, 6.313132)` | `(-7.248804, 4.040404)` |
| `label_three_mask1` | 1.543226e+00 | 9.576491e-01 | 4.068429e+05 | 4.104034e+02 | 93 | 11 | 1.182796e-01 | `(-8.395585, 4.755078)` | `(-7.248804, 4.040404)` |
| `label_nine_mask1` | 1.363245e+00 | 9.991091e-01 | 4.380795e+05 | 4.419532e+02 | 99 | 11 | 1.111111e-01 | `(-7.966507, 4.949495)` | `(-7.248804, 4.040404)` |
| `label_three_mask3` | 1.419029e+00 | 2.575264e+00 | 4.277714e+05 | 4.330835e+02 | 97 | 11 | 1.134021e-01 | `(-8.114487, 4.879725)` | `(-7.248804, 4.040404)` |

All four runs completed normally. No NaN/inf values were observed in parsed metrics.

## Best Runs

Best IoU:

```text
three_mask1, final defect_iou = 1.666667e-01
```

Area closest to the label:

```text
three_mask1, defect_area_pred = 66
```

The label-informed center runs have better centroids but worse IoU and larger predicted defect areas than `three_mask1`.

## Judgment

Combining label-informed centers with mask prior does not significantly improve the current framework. In this sample, oracle centers move the predicted centroid closer to the true defect, but they also expand the predicted low-`mu` area and reduce IoU relative to fixed `three` centers with mask prior.

This suggests that the current bottleneck is not just missing local information. Even with oracle center information and a mask prior, the weak-form + MuNet update tends to produce overly broad low-`mu` regions.

## Next Step

The next step should not be further prior tuning on this prototype. It is more useful to either:

- redesign the weak-form weights / quadrature and material update;
- revisit the MuNet parameterization or regularization;
- or stop this probe line and summarize it as evidence that the branch needs a more structured training design before scaling.
