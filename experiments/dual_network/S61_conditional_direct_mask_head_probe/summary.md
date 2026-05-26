# S61 conditional direct mask head probe

## Data source

- Reused S55 train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Reused S55 val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Reused S55 test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- Samples: train 1000, val 200, test 200.
- Resolution: 20x10.
- Execution device for completed S61 runs: `cuda`.

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
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

Compared configs:

| config | mask_head_mode | mask used for BCE / Dice and IoU |
|---|---|---|
| `mu_threshold_reference` | `mu_threshold` | `sigmoid((500 - mu) / mask_temperature)` for loss; `mu < 500` for metrics |
| `direct_mask_head` | `direct` | `mask_prob = sigmoid(mask_logits)` for loss; `mask_prob >= 0.5` for metrics |

## S61 average metrics

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `mu_threshold_reference` | train | 8.861030e-01 | 6.672000e+00 | 3.561543e+03 | 5.685660e+00 |
| `mu_threshold_reference` | val | 8.633440e-02 | 6.170000e+00 | 5.390944e+04 | 5.692302e+01 |
| `mu_threshold_reference` | test | 1.256267e-01 | 5.750000e+00 | 4.795601e+04 | 5.086337e+01 |
| `direct_mask_head` | train | 9.953174e-01 | 7.257000e+00 | 3.153239e+05 | 5.139315e+02 |
| `direct_mask_head` | val | 9.028429e-02 | 6.060000e+00 | 3.116588e+05 | 5.112933e+02 |
| `direct_mask_head` | test | 8.906158e-02 | 5.800000e+00 | 3.161132e+05 | 5.156228e+02 |

All metrics are finite and all train / eval / test CSV files record `mask_head_mode`.

## Comparison with S57 / S60

- S57 `norm_per_sample_zscore`: train / val / test IoU = `9.116529e-01` / `9.567763e-02` / `9.598926e-02`.
- S60 `no_local_signal_reference`: train / val / test IoU = `9.165308e-01` / `8.535813e-02` / `1.000321e-01`.
- S61 `mu_threshold_reference`: train / val / test IoU = `8.861030e-01` / `8.633440e-02` / `1.256267e-01`.
- S61 `direct_mask_head`: train / val / test IoU = `9.953174e-01` / `9.028429e-02` / `8.906158e-02`.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `mu_threshold_reference` | 7.997686e-01 | 7.604763e-01 |
| `direct_mask_head` | 9.050331e-01 | 9.062558e-01 |

## Current judgment

1. The direct mask head strongly improves train mask fitting, reaching near-perfect train IoU.
2. The direct head does not improve held-out test IoU; test IoU is lower than the S61 `mu_threshold_reference`.
3. The direct head only slightly improves val IoU over the S61 `mu_threshold_reference`, while increasing the train-to-held-out gap.
4. Because `lambda_mu_mse=0.0`, the direct head leaves the `mu` output effectively unoptimized; this is visible in the much larger `mu_mse` / `mu_mae`.
5. S61 suggests the main problem is not just the indirect `mu_pred -> mask` path. The bottleneck remains signal-to-shape generalization, supervision target design, loss balance, or dataset ambiguity.

## Next-step recommendation

- Do not replace the conditional baseline with direct mask head as-is.
- Keep `mu_threshold` as the current baseline for test comparison.
- If direct mask head is revisited, pair it with an explicit `mu` reconstruction term or multi-task balancing, then evaluate held-out performance rather than train IoU alone.
