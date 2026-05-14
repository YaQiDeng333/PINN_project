# S11_label_informed_centers

## Purpose

Run an oracle diagnostic with label-informed weak-form centers. S10 showed that global `abs(Bz)` peak placement is unreliable for this sample. S11 asks whether placing centers at the true `mu_label` defect centroid improves localization.

This is not a final unsupervised method. It uses label information only to diagnose whether center placement is the dominant bottleneck.

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

Label centroid:

```text
(-7.248804, 4.040404)
```

Compared center modes:

- `three`: fixed centers `[-5, 5]`, `[0, 5]`, `[5, 5]`
- `label_three`: oracle centers `[label_x - 2.5, label_y]`, `[label_x, label_y]`, `[label_x + 2.5, label_y]`
- `label_nine`: oracle `3 x 3` centers around `(label_x, label_y)`

## Results

| center_mode | final loss_phi | final loss_mu | final mu_mse | final mu_mae | defect_area_pred | defect_area_label | final defect_iou | pred centroid | label centroid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| three | 1.106874e+00 | 1.202469e-01 | 3.731120e+05 | 4.114994e+02 | 77 | 11 | 7.317073e-02 | `(-2.388927, 6.536796)` | `(-7.248804, 4.040404)` |
| label_three | 1.532790e+00 | 1.778343e-01 | 4.152714e+05 | 4.195043e+02 | 94 | 11 | 1.170213e-01 | `(-8.331467, 4.799055)` | `(-7.248804, 4.040404)` |
| label_nine | 1.265222e+00 | 2.047649e-01 | 4.443390e+05 | 4.479953e+02 | 100 | 11 | 1.100000e-01 | `(-7.894736, 5.000000)` | `(-7.248804, 4.040404)` |

All three runs completed normally. No NaN/inf values were observed in parsed metrics.

## Judgment

Label-informed oracle centers improve localization compared with fixed `three`:

- `label_three` increases IoU from `7.317073e-02` to `1.170213e-01`;
- `label_nine` increases IoU to `1.100000e-01`;
- both label-informed modes move the predicted centroid close to the label centroid.

However, the predicted defect area remains much too large:

- label area is `11`;
- `label_three` predicts `94`;
- `label_nine` predicts `100`.

This means center localization matters, but it is not sufficient. Even with oracle centers, the current weak-form + area-prior update still produces broad low-`mu` regions and weak IoU.

## Conclusion Split

The result does not fully support abandoning center selection. Oracle center placement helps centroid and IoU, so center localization is part of the bottleneck.

But the improvement is limited. Since label-informed centers still produce very large masks and IoU around `0.11`, the weak-form material update and area prior are not enough by themselves. The next direction should combine better center/weight design with a local or lightweight supervised diagnostic prior.
