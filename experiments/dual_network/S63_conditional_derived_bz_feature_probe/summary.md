# S63 conditional derived Bz signal feature probe

## Data source

- Reused S55 train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Reused S55 val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Reused S55 test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- Samples: train 1000, val 200, test 200.
- Resolution: 20x10.
- This is not a COMSOL multi-height experiment. It only derives richer features from the existing single Bz signal.

## Configuration

Shared configuration:

- Steps: 5000
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `encoder_type=mlp`
- `conditioning_mode=concat`
- `signal_normalization=per_sample_zscore`
- `point_signal_mode=none`
- `mask_head_mode=mu_threshold`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

Compared signal feature modes:

| config | signal_feature_mode | encoder input length |
|---|---|---:|
| `raw_reference` | `raw` | 20 |
| `raw_abs_grad` | `raw_abs_grad` | 60 |

`raw_abs_grad` concatenates normalized raw Bz, `abs(Bz)`, and a finite-difference gradient along the signal axis.

## S63 average metrics

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `raw_reference` | train | 8.758866e-01 | 6.623000e+00 | 3.759930e+03 | 6.222000e+00 |
| `raw_reference` | val | 9.413832e-02 | 5.620000e+00 | 5.024390e+04 | 5.388258e+01 |
| `raw_reference` | test | 1.040388e-01 | 5.170000e+00 | 4.725346e+04 | 5.053412e+01 |
| `raw_abs_grad` | train | 9.142336e-01 | 6.860000e+00 | 2.668627e+03 | 5.564395e+00 |
| `raw_abs_grad` | val | 1.081610e-01 | 5.795000e+00 | 4.988274e+04 | 5.356188e+01 |
| `raw_abs_grad` | test | 1.036962e-01 | 6.200000e+00 | 5.154910e+04 | 5.552483e+01 |

All metrics are finite and train / eval / test CSV files were generated for both configs.

## Comparison with S57 / S60 references

- S57 `per_sample_zscore`: train / val / test IoU = `9.116529e-01` / `9.567763e-02` / `9.598926e-02`.
- S60 `no_local_signal_reference`: train / val / test IoU = `9.166308e-01` / `8.535813e-02` / `1.000321e-01`.
- S63 `raw_reference`: train / val / test IoU = `8.758866e-01` / `9.413832e-02` / `1.040388e-01`.
- S63 `raw_abs_grad`: train / val / test IoU = `9.142336e-01` / `1.081610e-01` / `1.036962e-01`.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `raw_reference` | 7.817483e-01 | 7.718478e-01 |
| `raw_abs_grad` | 8.060726e-01 | 8.105374e-01 |

## Current judgment

1. `raw_abs_grad` improves train IoU and val IoU relative to `raw_reference`.
2. `raw_abs_grad` does not improve test IoU relative to `raw_reference`; the test difference is essentially flat to slightly worse.
3. Derived single-signal features therefore do not provide a stable held-out generalization improvement.
4. The remaining bottleneck is more likely single-Bz information limits, signal-to-shape ambiguity, dataset design, or supervision / loss design than simply missing `abs` or finite-difference signal channels.

## Next-step recommendation

- Keep `raw` / `mu_threshold` as the conditional baseline reference unless a later validation policy chooses otherwise.
- Do not keep tuning single-signal derived features blindly.
- If the branch continues, the next diagnostic should move toward a multi-height / COMSOL Bz data interface or a clearer signal-to-shape dataset design, rather than only adding derived channels to the current single Bz input.
