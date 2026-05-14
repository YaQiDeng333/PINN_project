# S20 Runner 20-Sample 40x20 BCE Probe

## Data Generation

Command:

```powershell
python data_generator_v2.py --train-samples 20 --val-samples 0 --test-samples 0 --grid-x 40 --grid-y 20 --output-dir experiments\dual_network\S20_runner_20sample_40x20_bce_probe\data --seed 1020
```

Generated train data:

- `experiments/dual_network/S20_runner_20sample_40x20_bce_probe/data/training_data_train.npz`

## Runner Configuration

Both runs used:

- `grid_x=40`, `grid_y=20`
- `sample_indices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- `outer_steps=20`
- `phi_steps=20`
- `mu_steps=20`
- `test_radius=5.0`
- `center_mode=three`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `area_prior_temperature=50.0`
- `mask_prior_temperature=50.0`

Compared runs:

- baseline: `lambda_mask_bce_prior=0.0`
- bce: `lambda_mask_bce_prior=1.0`

S20 is a resolution feasibility probe. It uses a 40x20 grid and fewer 20/20/20 training steps, so it is not a strict numerical comparison against S19's 20x10 grid with 30/30/30 steps.

## Twenty-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.077682e-01 | 2.858000e+02 | 3.107496e+05 | 3.890810e+02 |
| bce | 1.739786e-01 | 2.125500e+02 | 2.053370e+05 | 3.794331e+02 |

## Per-Sample IoU and Area

| sample | baseline IoU | bce IoU | baseline area | bce area | label area |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.322259e-02 | 6.415094e-02 | 291 | 262 | 20 |
| 1 | 5.517241e-02 | 8.695652e-02 | 286 | 230 | 20 |
| 2 | 5.940594e-02 | 6.691450e-02 | 300 | 266 | 21 |
| 3 | 8.487085e-02 | 8.296943e-02 | 271 | 225 | 23 |
| 4 | 3.921569e-02 | 5.172414e-02 | 306 | 232 | 12 |
| 5 | 2.230483e-01 | 4.000000e-01 | 267 | 155 | 62 |
| 6 | 1.603261e-01 | 6.818182e-01 | 366 | 87 | 61 |
| 7 | 2.160000e-01 | 2.571429e-01 | 250 | 210 | 54 |
| 8 | 1.718213e-01 | 3.544304e-01 | 285 | 158 | 56 |
| 9 | 7.116105e-02 | 9.268293e-02 | 267 | 205 | 19 |
| 10 | 7.954545e-02 | 9.210526e-02 | 264 | 228 | 21 |
| 11 | 2.864583e-02 | 4.633205e-02 | 376 | 252 | 19 |
| 12 | 8.633094e-02 | 1.111111e-01 | 272 | 250 | 30 |
| 13 | 1.434263e-01 | 1.548117e-01 | 250 | 239 | 37 |
| 14 | 8.214286e-02 | 1.290323e-01 | 279 | 186 | 24 |
| 15 | 5.295950e-02 | 7.659575e-02 | 320 | 235 | 18 |
| 16 | 1.397059e-01 | 1.759259e-01 | 272 | 216 | 38 |
| 17 | 1.297710e-01 | 1.931818e-01 | 262 | 176 | 34 |
| 18 | 1.686275e-01 | 2.037915e-01 | 255 | 211 | 43 |
| 19 | 1.299639e-01 | 1.578947e-01 | 277 | 228 | 36 |

## Stability Check

- BCE improves average IoU from `1.077682e-01` to `1.739786e-01`.
- BCE improves IoU on 19 of 20 samples; sample 3 has a slight IoU regression from `8.487085e-02` to `8.296943e-02`.
- BCE reduces average predicted defect area from `285.8` to `212.55`, and the predicted area is closer to label area on all 20 samples.
- BCE lowers `mu_mse` on all 20 samples.
- BCE lowers `mu_mae` on 9 of 20 samples; the average `mu_mae` is still slightly better than baseline.
- No NaN/inf was observed in metrics or run logs.
- `metrics.csv` was generated for both baseline and BCE runs.

## Failure Samples

The high-resolution run is materially harder than S19. BCE remains better on average, but the final masks are still too large for many samples.

Notable weak cases:

- sample 3: BCE IoU slightly regresses to `8.296943e-02`;
- sample 11: BCE IoU is only `4.633205e-02`, with predicted area 252 vs label area 19;
- sample 0, 1, 2, 4, 9, 10, 12, 15: BCE IoU remains below `1.2e-01`;
- sample 19: BCE initially improves during training but ends with area expansion and final IoU `1.578947e-01`.

## Current Conclusion

S20 shows that BCE mask prior still provides a semi-supervised signal at 40x20 resolution: averages improve, predicted area is closer to the label area, and `mu_mse` improves on all 20 samples.

However, the improvement is much weaker than S18/S19. With only 20/20/20 steps on the 40x20 grid, BCE does not deliver robust localization across all samples, and many predictions remain over-expanded. This is a feasibility signal for the semi-supervised route, not a resolution-robust result yet.

This remains a semi-supervised result. BCE uses `mu_label < 500`, so S20 does not prove unsupervised weak-form inversion success.
