# S58 conditional FiLM conditioning probe

## 数据来源

- Reused S55 train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Reused S55 val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Reused S55 test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- Samples: train 1000, val 200, test 200.
- Resolution: 20x10.

## 配置

All configs use:

- Steps: 5000
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

Compared configs:

| config | conditioning_mode | signal_normalization |
|---|---|---|
| `concat_per_sample_zscore_reference` | `concat` | `per_sample_zscore` |
| `film_per_sample_zscore` | `film` | `per_sample_zscore` |
| `film_train_zscore` | `film` | `train_zscore` |

## S58 平均指标

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `concat_per_sample_zscore_reference` | train | 8.649705e-01 | 6.595000e+00 | 3.938099e+03 | 6.375763e+00 |
| `concat_per_sample_zscore_reference` | val | 7.280748e-02 | 5.620000e+00 | 5.229190e+04 | 5.574832e+01 |
| `concat_per_sample_zscore_reference` | test | 1.063626e-01 | 5.380000e+00 | 4.854544e+04 | 5.206356e+01 |
| `film_per_sample_zscore` | train | 9.563841e-01 | 7.046000e+00 | 1.599733e+03 | 8.233177e+00 |
| `film_per_sample_zscore` | val | 6.837561e-02 | 5.450000e+00 | 4.582150e+04 | 6.059949e+01 |
| `film_per_sample_zscore` | test | 7.330513e-02 | 4.720000e+00 | 4.386645e+04 | 5.903585e+01 |
| `film_train_zscore` | train | 9.696210e-01 | 7.146000e+00 | 1.252331e+03 | 7.493121e+00 |
| `film_train_zscore` | val | 8.486297e-02 | 4.625000e+00 | 4.326102e+04 | 5.372972e+01 |
| `film_train_zscore` | test | 8.279666e-02 | 5.490000e+00 | 4.550577e+04 | 5.815795e+01 |

All metrics are finite and all train / eval / test CSV files record `conditioning_mode` and `signal_normalization`.

## 与 S57 per_sample_zscore 对比

- S57 `norm_per_sample_zscore`: train / val / test IoU = `9.116529e-01` / `9.567763e-02` / `9.598926e-02`.
- S58 `concat_per_sample_zscore_reference`: train / val / test IoU = `8.649705e-01` / `7.280748e-02` / `1.063626e-01`.
- S58 `film_per_sample_zscore`: train / val / test IoU = `9.563841e-01` / `6.837561e-02` / `7.330513e-02`.
- S58 `film_train_zscore`: train / val / test IoU = `9.696210e-01` / `8.486297e-02` / `8.279666e-02`.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `concat_per_sample_zscore_reference` | 7.921630e-01 | 7.586079e-01 |
| `film_per_sample_zscore` | 8.880085e-01 | 8.830789e-01 |
| `film_train_zscore` | 8.847580e-01 | 8.868243e-01 |

## 当前判断

1. FiLM conditioning strongly improves train fitting and continuous `mu_mse`, especially `film_train_zscore`.
2. FiLM does not improve held-out IoU in S58. Both FiLM configs have lower test IoU than the concat per-sample-zscore reference, and `film_per_sample_zscore` also has lower val IoU.
3. FiLM currently increases overfitting rather than solving conditional generalization.
4. The best S58 held-out IoU is mixed: concat has the best test IoU, while `film_train_zscore` has the best val IoU and lower continuous errors. None is strong enough for main-baseline comparison.

## 下一步建议

- Do not treat FiLM as the new default conditioning mode.
- If continuing conditional work, prioritize signal encoder architecture, regularization, validation-aware early stopping, and loss design.
- FiLM can remain a diagnostic branch for train fitting, but not the current generalization candidate.
