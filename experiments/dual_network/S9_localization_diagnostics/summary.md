# S9_localization_diagnostics

## Purpose

Diagnose the localization failure exposed by S7/S8. The goal is not to tune another scalar hyperparameter, but to save final `mu_pred`, `mu_label`, threshold masks, centroids, and images for direct inspection.

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
center_mode = three
area_prior_temperature = 50.0
```

Compared runs:

- `lambda_area_prior = 0.0`
- `lambda_area_prior = 1.0`

## Saved Diagnostics

Each run directory contains:

- `run_stdout.txt`
- `final_mu_pred.npy`
- `final_mu_label.npy`
- `final_pred_mask.npy`
- `final_label_mask.npy`
- `final_diagnostics.txt`
- `mu_pred_vs_label.png`

Run directories:

- `experiments/dual_network/S9_localization_diagnostics/lambda_area_0/`
- `experiments/dual_network/S9_localization_diagnostics/lambda_area_1/`

## Final Metrics

| lambda_area_prior | final loss_phi | final loss_mu | final mu_mse | final mu_mae | defect_area_pred | defect_area_label | final defect_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1.417249e-01 | 1.033289e-03 | 9.337209e+05 | 9.392122e+02 | 200 | 11 | 5.500000e-02 |
| 1.0 | 1.106874e+00 | 1.202469e-01 | 3.731120e+05 | 4.114994e+02 | 77 | 11 | 7.317073e-02 |

## Centroids

| lambda_area_prior | pred centroid | label centroid |
| --- | --- | --- |
| 0.0 | `(-3.814697e-08, 5.000000e+00)` | `(-7.248804e+00, 4.040404e+00)` |
| 1.0 | `(-2.388927e+00, 6.536796e+00)` | `(-7.248804e+00, 4.040404e+00)` |

## Image-Based Diagnosis

For `lambda_area_prior=0.0`, the predicted mask is the whole `20 x 10` domain. This is a full-domain low-`mu` collapse, not a localized defect prediction.

For `lambda_area_prior=1.0`, the area prior reduces the predicted low-`mu` region from 200 points to 77 points, but the predicted mask is still too large and shifted. Its centroid is far to the right and above the label centroid. The image shows a broad, curved low-`mu` band rather than a compact defect matching the label.

The failure mode is therefore:

- no prior: full-domain low-`mu` collapse;
- lambda 1.0: area reduced, but still overlarge with position shift and shape mismatch.

The IoU only improves from `5.500000e-02` to `7.317073e-02`, so the area prior helps suppress collapse but does not solve localization.

## Next Step

If this branch continues, the next step should not be simply increasing the area prior. S8 already showed that stronger area priors reduce area while IoU drops to zero. The next useful direction is to add localization information through weak-form weights, better test-function placement, a local `mu` prior, or a lightweight supervised diagnostic prior.
