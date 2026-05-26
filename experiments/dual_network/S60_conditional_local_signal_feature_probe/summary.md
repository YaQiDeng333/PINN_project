# S60 conditional local signal feature probe

## Data source

- Reused S55 train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Reused S55 val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Reused S55 test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- Samples: train 1000, val 200, test 200.
- Resolution: 20x10.
- Execution device for completed S60 runs: `cuda`.

## Configuration

All configs use:

- Steps: 5000
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `encoder_type=mlp`
- `conditioning_mode=concat`
- `signal_normalization=per_sample_zscore`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

Compared configs:

| config | point_signal_mode | extra point features |
|---|---|---|
| `no_local_signal_reference` | `none` | none |
| `local_value` | `local_value` | local Bz value aligned by x coordinate |
| `local_value_abs` | `local_value_abs` | local Bz value and absolute local Bz value |

## S60 average metrics

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `no_local_signal_reference` | train | 9.165308e-01 | 6.877000e+00 | 2.351577e+03 | 4.433288e+00 |
| `no_local_signal_reference` | val | 8.535813e-02 | 5.245000e+00 | 5.008374e+04 | 5.327860e+01 |
| `no_local_signal_reference` | test | 1.000321e-01 | 5.495000e+00 | 4.853060e+04 | 5.181712e+01 |
| `local_value` | train | 8.968483e-01 | 6.740000e+00 | 2.983464e+03 | 4.911460e+00 |
| `local_value` | val | 8.595360e-02 | 5.270000e+00 | 4.875788e+04 | 5.194455e+01 |
| `local_value` | test | 9.525210e-02 | 5.650000e+00 | 4.934096e+04 | 5.273595e+01 |
| `local_value_abs` | train | 8.831255e-01 | 6.676000e+00 | 3.464424e+03 | 5.626067e+00 |
| `local_value_abs` | val | 9.574179e-02 | 5.590000e+00 | 4.930176e+04 | 5.255689e+01 |
| `local_value_abs` | test | 9.469099e-02 | 5.110000e+00 | 4.738315e+04 | 5.046165e+01 |

All metrics are finite and all train / eval / test CSV files record `point_signal_mode`.

## Comparison with S57 / S59

- S57 `norm_per_sample_zscore`: train / val / test IoU = `9.116529e-01` / `9.567763e-02` / `9.598926e-02`.
- S59 `mlp_concat_reference`: train / val / test IoU = `8.432535e-01` / `9.821131e-02` / `1.118232e-01`.
- S60 `no_local_signal_reference`: train / val / test IoU = `9.165308e-01` / `8.535813e-02` / `1.000321e-01`.
- S60 `local_value`: train / val / test IoU = `8.968483e-01` / `8.595360e-02` / `9.525210e-02`.
- S60 `local_value_abs`: train / val / test IoU = `8.831255e-01` / `9.574179e-02` / `9.469099e-02`.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `no_local_signal_reference` | 8.311727e-01 | 8.164987e-01 |
| `local_value` | 8.108947e-01 | 8.015962e-01 |
| `local_value_abs` | 7.873837e-01 | 7.884345e-01 |

## Current judgment

1. `local_value` does not improve held-out IoU relative to the no-local reference.
2. `local_value_abs` gives the best S60 val IoU and slightly lower test `mu_mse` / `mu_mae`, but its test IoU is below the no-local reference.
3. Coordinate-aligned local Bz features do not provide a stable val/test IoU improvement in this setup.
4. The weak held-out generalization is therefore not explained solely by global latent compression of local signal information.
5. If continuing local-signal diagnostics, `local_value_abs` is the more useful diagnostic mode, but it should not replace the no-local baseline as a default.

## Next-step recommendation

- Do not treat local signal features as a solved conditional generalization fix.
- Keep the no-local MLP concat baseline for held-out test comparison.
- If continuing the conditional path, prioritize validation-aware training, regularization, target / loss design, or dataset ambiguity checks before adding more pointwise signal features.
