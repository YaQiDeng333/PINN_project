# S10_signal_informed_centers

## Purpose

Test signal-informed compact-support weak-form centers. Instead of using only fixed centers, the loop finds the probe-line location with the largest absolute `Bz` response and places centers around that `x_peak`.

This checks whether centering weak-form tests near the signal peak improves the localization failure observed in S9.

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
```

Signal peak:

```text
x_peak = 1.184210e+01
```

Compared center modes:

- `three`: fixed centers `[-5, 5]`, `[0, 5]`, `[5, 5]`
- `signal_three`: centers around `x_peak`: `[x_peak - 2.5, 5]`, `[x_peak, 5]`, `[x_peak + 2.5, 5]`, clamped to `[-15, 15]`
- `signal_nine`: `3 x 3` centers using `x = [x_peak - 2.5, x_peak, x_peak + 2.5]` and `y = [2.5, 5.0, 7.5]`, clamped to `[-15, 15]`

## Results

| center_mode | final loss_phi | final loss_mu | final mu_mse | final mu_mae | defect_area_pred | defect_area_label | final defect_iou | pred centroid | label centroid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| three | 1.106874e+00 | 1.202469e-01 | 3.731120e+05 | 4.114994e+02 | 77 | 11 | 7.317073e-02 | `(-2.388927, 6.536796)` | `(-7.248804, 4.040404)` |
| signal_three | 1.816097e+00 | 1.278194e-01 | 4.443060e+05 | 4.481765e+02 | 78 | 11 | 0.000000e+00 | `(9.493926, 4.544160)` | `(-7.248804, 4.040404)` |
| signal_nine | 1.827116e+00 | 1.267072e-01 | 4.391736e+05 | 4.424220e+02 | 77 | 11 | 0.000000e+00 | `(9.586466, 4.559885)` | `(-7.248804, 4.040404)` |

All runs completed normally. No NaN/inf values were observed in parsed metrics.

## Judgment

Signal-informed centers did not improve localization in this sample.

The detected `x_peak` is near `11.84`, while the label centroid is near `x=-7.25`. Placing weak-form centers around the signal peak shifts the predicted low-`mu` centroid to the far right, around `x=9.5`, and removes overlap with the label mask. Both `signal_three` and `signal_nine` end with zero IoU.

The fixed `three` center mode remains better for this single sample because it is the only tested setting with nonzero final IoU.

## Current Limit

This result only applies to one generated sample. The current analytic signal can have a peak that does not directly correspond to the true defect centroid, so naive `argmax(abs(Bz))` is not a reliable localization rule here.

## Next Step

If this branch continues, avoid using raw global `abs(Bz)` argmax as the sole center placement rule. The next localization attempt should use a more robust signal feature, local weak-form weights, or a lightweight supervised/local prior.
