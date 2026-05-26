# S62 conditional direct mask multi-task loss probe

## Data source

- Reused S55 train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Reused S55 val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Reused S55 test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- Samples: train 1000, val 200, test 200.
- Resolution: 20x10.
- Execution device for completed S62 runs: `cuda`.

## Configuration

All configs use:

- Steps: 5000
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `encoder_type=mlp`
- `conditioning_mode=concat`
- `signal_normalization=per_sample_zscore`
- `point_signal_mode=none`
- `mask_head_mode=direct`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `mask_temperature=50.0`

Compared configs:

| config | lambda_mu_mse |
|---|---:|
| `direct_mu0_reference` | 0.0 |
| `direct_mu1e-5` | 1e-5 |
| `direct_mu1e-4` | 1e-4 |

## S62 average metrics

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `direct_mu0_reference` | train | 9.828120e-01 | 7.191000e+00 | 2.385354e+05 | 4.742066e+02 |
| `direct_mu0_reference` | val | 1.051470e-01 | 6.410000e+00 | 2.380895e+05 | 4.734551e+02 |
| `direct_mu0_reference` | test | 9.803234e-02 | 5.930000e+00 | 2.335287e+05 | 4.689150e+02 |
| `direct_mu1e-5` | train | 9.937419e-01 | 7.254000e+00 | 3.643501e+03 | 3.909185e+00 |
| `direct_mu1e-5` | val | 9.816303e-02 | 6.585000e+00 | 5.443113e+04 | 5.573441e+01 |
| `direct_mu1e-5` | test | 9.670001e-02 | 6.000000e+00 | 5.399731e+04 | 5.539699e+01 |
| `direct_mu1e-4` | train | 9.199159e-01 | 7.156000e+00 | 3.605718e+03 | 4.569292e+00 |
| `direct_mu1e-4` | val | 1.003297e-01 | 5.920000e+00 | 5.112492e+04 | 5.305583e+01 |
| `direct_mu1e-4` | test | 8.550005e-02 | 6.050000e+00 | 5.202169e+04 | 5.378262e+01 |

All metrics are finite and all train / eval / test CSV files record `mask_head_mode=direct`.

## Comparison with S61

- S61 `mu_threshold_reference`: train / val / test IoU = `8.861030e-01` / `8.633440e-02` / `1.256267e-01`; test `mu_mse=4.795601e+04`, test `mu_mae=5.086337e+01`.
- S61 `direct_mask_head`: train / val / test IoU = `9.953174e-01` / `9.028429e-02` / `8.906158e-02`; test `mu_mse=3.161132e+05`, test `mu_mae=5.156228e+02`.
- S62 `direct_mu1e-5` keeps high train IoU and reduces test `mu_mse` / `mu_mae` to the same order as the `mu_threshold` baseline, but test IoU remains below S61 `mu_threshold_reference`.
- S62 `direct_mu1e-4` further regularizes `mu` but lowers train and test IoU relative to `direct_mu1e-5`.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `direct_mu0_reference` | 8.776650e-01 | 8.847797e-01 |
| `direct_mu1e-5` | 8.955789e-01 | 8.970419e-01 |
| `direct_mu1e-4` | 8.195862e-01 | 8.344159e-01 |

## Current judgment

1. Adding `lambda_mu_mse` strongly improves the direct head's continuous `mu` error.
2. `direct_mu1e-5` preserves near-perfect train mask IoU while reducing `mu_mse` / `mu_mae` by roughly two orders of magnitude compared with `direct_mu0_reference`.
3. The `mu_mse` term does not improve held-out test IoU; all direct-head S62 configs remain below the S61 `mu_threshold_reference` test IoU.
4. `direct_mu1e-4` suggests stronger `mu_mse` creates a task trade-off: better continuous `mu` discipline but lower mask IoU.
5. Direct multi-task is useful diagnostically, but it is not yet a better conditional baseline.

## Next-step recommendation

- Keep `mu_threshold_reference` as the conditional held-out baseline.
- If direct mask head remains in the branch, use `direct_mu1e-5` as the best diagnostic multi-task setting.
- Do not continue optimizing only train IoU; next work should focus on held-out signal-to-shape generalization, validation-aware selection, and dataset ambiguity.
