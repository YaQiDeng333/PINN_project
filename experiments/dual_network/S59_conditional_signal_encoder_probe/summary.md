# S59 conditional signal encoder architecture probe

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
- `signal_normalization=per_sample_zscore`

Compared configs:

| config | encoder_type | conditioning_mode |
|---|---|---|
| `mlp_concat_reference` | `mlp` | `concat` |
| `cnn_concat` | `cnn` | `concat` |
| `cnn_film` | `cnn` | `film` |

## S59 平均指标

| config | split | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
|---|---|---:|---:|---:|---:|
| `mlp_concat_reference` | train | 8.432535e-01 | 6.455000e+00 | 4.822165e+03 | 7.406344e+00 |
| `mlp_concat_reference` | val | 9.821131e-02 | 5.810000e+00 | 5.064852e+04 | 5.452025e+01 |
| `mlp_concat_reference` | test | 1.118232e-01 | 5.795000e+00 | 4.875783e+04 | 5.274552e+01 |
| `cnn_concat` | train | 9.442416e-01 | 7.003000e+00 | 3.039078e+03 | 1.260097e+01 |
| `cnn_concat` | val | 9.915262e-02 | 5.925000e+00 | 5.032018e+04 | 5.908413e+01 |
| `cnn_concat` | test | 9.681660e-02 | 5.905000e+00 | 5.113028e+04 | 6.003907e+01 |
| `cnn_film` | train | 9.844595e-01 | 7.206000e+00 | 4.027536e+02 | 2.369864e+00 |
| `cnn_film` | val | 1.254790e-01 | 5.895000e+00 | 4.801250e+04 | 5.252832e+01 |
| `cnn_film` | test | 1.094247e-01 | 5.520000e+00 | 4.869030e+04 | 5.317442e+01 |

All metrics are finite and all train / eval / test CSV files record `encoder_type`, `conditioning_mode`, and `signal_normalization`.

## 与 S57 / S58 对比

- S57 `norm_per_sample_zscore`: train / val / test IoU = `9.116529e-01` / `9.567763e-02` / `9.598926e-02`.
- S58 `concat_per_sample_zscore_reference`: train / val / test IoU = `8.649705e-01` / `7.280748e-02` / `1.063626e-01`.
- S59 `mlp_concat_reference`: train / val / test IoU = `8.432535e-01` / `9.821131e-02` / `1.118232e-01`.
- S59 `cnn_concat`: train / val / test IoU = `9.442416e-01` / `9.915262e-02` / `9.681660e-02`.
- S59 `cnn_film`: train / val / test IoU = `9.844595e-01` / `1.254790e-01` / `1.094247e-01`.

## Train-val / train-test gap

| config | train-val IoU gap | train-test IoU gap |
|---|---:|---:|
| `mlp_concat_reference` | 7.450422e-01 | 7.314303e-01 |
| `cnn_concat` | 8.450889e-01 | 8.474250e-01 |
| `cnn_film` | 8.589805e-01 | 8.750348e-01 |

## 当前判断

1. `cnn_concat` improves train fitting relative to `mlp_concat_reference`, but does not improve test IoU and has worse continuous held-out errors.
2. `cnn_film` gives the highest train IoU, lowest train `mu` error, and best val IoU in S59, but it also has the largest train-to-held-out gap.
3. CNN encoder helps representation capacity, but the held-out gains are modest and not uniformly better than MLP concat.
4. CNN + FiLM is better than CNN + concat on val IoU and continuous errors, but still looks overfit; it is not a clear generalization solution.

## 下一步建议

- Do not treat CNN encoder or CNN + FiLM as a solved default.
- If continuing conditional work, use `cnn_film` as a diagnostic high-capacity setting and `mlp_concat_reference` as the generalization baseline.
- The next bottleneck is likely validation-aware training, regularization, loss design, or dataset ambiguity rather than encoder structure alone.
