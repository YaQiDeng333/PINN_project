# S16 Train Runner Probe

## Data

- npz: `experiments/dual_network/S15_multi_sample_bce_probe/data/training_data_train.npz`
- samples: `0,1,2`
- grid: 20 x 10

## Runner Configuration

Both runs used:

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

## Output Files

- baseline metrics: `experiments/dual_network/S16_train_runner_probe/baseline/metrics.csv`
- bce metrics: `experiments/dual_network/S16_train_runner_probe/bce/metrics.csv`
- each sample directory contains `run_log.txt`, final diagnostics, final maps, masks, and `mu_pred_vs_label.png`

## Three-Sample Averages

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 9.287478e-02 | 6.400000e+01 | 2.867328e+05 | 3.224674e+02 |
| bce | 7.840909e-01 | 8.000000e+00 | 1.597866e+04 | 6.105250e+01 |

## Per-Sample Final Metrics

| sample | run | defect_area_pred | defect_area_label | defect_iou | mu_mse | mu_mae |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | baseline | 75 | 8 | 1.066667e-01 | 3.315026e+05 | 3.582430e+02 |
| 0 | bce | 11 | 8 | 7.272727e-01 | 1.991005e+04 | 6.592333e+01 |
| 1 | baseline | 54 | 5 | 9.259259e-02 | 2.416857e+05 | 2.955117e+02 |
| 1 | bce | 8 | 5 | 6.250000e-01 | 2.236068e+04 | 8.266766e+01 |
| 2 | baseline | 63 | 5 | 7.936508e-02 | 2.870101e+05 | 3.136474e+02 |
| 2 | bce | 5 | 5 | 1.000000e+00 | 5.665255e+03 | 3.456652e+01 |

## Interpretation

The runner reproduces the S15 conclusion: the BCE mask prior consistently improves area control, IoU, and material-map error on the same 3-sample dataset.

`train_dual_variational.py` is now usable as a small-scale branch experiment runner for repeated .npz sample sweeps. It is still not a formal large-scale training pipeline and does not save model weights.

## Limitations

- BCE and mask priors use `mu_label`, so the BCE result is a semi-supervised diagnostic upper bound.
- The experiment uses only 3 samples on a 20 x 10 grid.
- The runner trains one independent PhiNet/MuNet pair per sample and does not implement batch training.
- No model checkpoints are saved.
