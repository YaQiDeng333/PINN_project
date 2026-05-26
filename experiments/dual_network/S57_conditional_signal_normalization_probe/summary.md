# S57 conditional signal normalization probe

## 数据来源

- Reused S55 train NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_train.npz`
- Reused S55 val NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_val.npz`
- Reused S55 test NPZ: `experiments/dual_network/S55_conditional_datascale_generalization_probe/data/training_data_test.npz`
- Samples: train 1000, val 200, test 200.
- Resolution: 20x10.

## 配置

All configs use the S55/S56 big BCE + Dice setup:

- Steps: 5000
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`

Compared signal normalization modes:

| config | signal_normalization |
|---|---|
| `norm_none_reference` | `none` |
| `norm_train_zscore` | `train_zscore` |
| `norm_per_sample_zscore` | `per_sample_zscore` |

## S57 平均指标

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `norm_none_reference` | train | 9.114212e-01 | 6.801000e+00 | 3.037045e+03 | 7.055326e+00 |
| `norm_none_reference` | val | 9.318296e-02 | 5.595000e+00 | 4.829483e+04 | 5.361569e+01 |
| `norm_none_reference` | test | 8.784143e-02 | 5.480000e+00 | 4.871150e+04 | 5.379750e+01 |
| `norm_train_zscore` | train | 8.914478e-01 | 6.706000e+00 | 3.273402e+03 | 5.324528e+00 |
| `norm_train_zscore` | val | 7.994475e-02 | 5.930000e+00 | 5.250681e+04 | 5.600090e+01 |
| `norm_train_zscore` | test | 8.733596e-02 | 5.565000e+00 | 5.105319e+04 | 5.435084e+01 |
| `norm_per_sample_zscore` | train | 9.116529e-01 | 6.809000e+00 | 2.735445e+03 | 5.111933e+00 |
| `norm_per_sample_zscore` | val | 9.567763e-02 | 6.330000e+00 | 5.305011e+04 | 5.713012e+01 |
| `norm_per_sample_zscore` | test | 9.598926e-02 | 5.760000e+00 | 5.030556e+04 | 5.420592e+01 |

All metrics are finite and all train / eval / test CSV files record `signal_normalization`.

## 与 S55 / S56 big 配置对比

- S55 `big_bce_dice_datascale`: train / val / test IoU = `9.130737e-01` / `8.848209e-02` / `7.708859e-02`.
- S56 `big_bce_dice_signal_ablation` correct signal: train / val / test IoU = `9.072941e-01` / `9.372990e-02` / `8.830051e-02`.
- S57 `norm_none_reference`: train / val / test IoU = `9.114212e-01` / `9.318296e-02` / `8.784143e-02`.
- S57 `norm_train_zscore`: train / val / test IoU = `8.914478e-01` / `7.994475e-02` / `8.733596e-02`.
- S57 `norm_per_sample_zscore`: train / val / test IoU = `9.116529e-01` / `9.567763e-02` / `9.598926e-02`.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `norm_none_reference` | 8.182383e-01 | 8.235797e-01 |
| `norm_train_zscore` | 8.115030e-01 | 8.041119e-01 |
| `norm_per_sample_zscore` | 8.159753e-01 | 8.156637e-01 |

## 当前判断

1. `train_zscore` does not improve val/test IoU; it slightly lowers val IoU and leaves test IoU close to `none`.
2. `per_sample_zscore` gives the best val/test IoU in S57 and the best train continuous `mu` errors, but the gain is modest and held-out IoU remains low.
3. Simple raw signal scale is not the main bottleneck. If normalization helps, it is only a small effect.
4. The conditional generalization problem still points more strongly to encoder / conditioning architecture, regularization, and loss design than to simple signal scaling.

## 下一步建议

- Continue with `per_sample_zscore` only as a weakly better diagnostic setting, not as a solved default.
- Prioritize signal encoder / conditioning changes over more scalar normalization sweeps.
- Do not compare against the main baseline yet, because held-out IoU remains low at 20x10.
