# S105 COMSOL V2 threshold-margin probe

## 目的

S105 测试 threshold-margin loss 是否能让 V2 正样本 `mu_pred` 跨过 `mu_threshold=500`，从而恢复非零 hard mask / IoU。

## 数据和共同配置

- train NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- train / val / test samples: `100 / 20 / 20`
- flattened signal length: `600`
- `hidden_dim=128`, `num_layers=4`, `latent_dim=64`
- `steps=3000`, `lr=1e-3`
- `mask_head_mode=mu_threshold`
- `mask_source=mu_threshold`
- `train_point_subsample=4096`
- `history_interval=250`

## 结果

| config | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| v2_baseline_reference | train | 0.000000e+00 | 0.000000e+00 | 1.535943e+05 | 3.878448e+02 |
| v2_baseline_reference | val | 0.000000e+00 | 0.000000e+00 | 1.536635e+05 | 3.879139e+02 |
| v2_baseline_reference | test | 0.000000e+00 | 0.000000e+00 | 1.535795e+05 | 3.878299e+02 |
| positive_margin_lambda1 | train | 5.355850e-02 | 2.000000e+04 | 9.440432e+05 | 9.452559e+02 |
| positive_margin_lambda1 | val | 5.383750e-02 | 2.000000e+04 | 9.437644e+05 | 9.449771e+02 |
| positive_margin_lambda1 | test | 5.350000e-02 | 2.000000e+04 | 9.441090e+05 | 9.453178e+02 |
| positive_margin_lambda10 | train | 5.355850e-02 | 2.000000e+04 | 9.417171e+05 | 9.441566e+02 |
| positive_margin_lambda10 | val | 5.383750e-02 | 2.000000e+04 | 9.414516e+05 | 9.438845e+02 |
| positive_margin_lambda10 | test | 5.350000e-02 | 2.000000e+04 | 9.417943e+05 | 9.442239e+02 |
| bidirectional_margin_lambda1 | train | 1.305192e-01 | 7.883350e+03 | 2.222390e+05 | 4.685095e+02 |
| bidirectional_margin_lambda1 | val | 1.185913e-01 | 8.412650e+03 | 2.275314e+05 | 4.746521e+02 |
| bidirectional_margin_lambda1 | test | 1.266814e-01 | 8.074100e+03 | 2.235250e+05 | 4.698980e+02 |

## training history 观察

- `v2_baseline_reference` 最后 `sampled_mu_positive_mean=625.94`，`batch_area_pred=0`，仍未跨过 threshold。
- `positive_margin_lambda1` 将正样本 `sampled_mu_positive_mean` 压到约 `1.27`，但同时负样本也被压到约 `1.27`，导致全前景预测。
- `positive_margin_lambda10` 现象相同，最终 hard area 为整张图 `20000`。
- `bidirectional_margin_lambda1` 最终 `sampled_mu_positive_mean=482.84`，`sampled_mu_negative_mean=530.01`，说明正负 margin 都在起作用；hard area 非零且不是全图。

## 当前判断

- threshold-margin loss 能恢复非零 hard mask / IoU。
- positive-only margin 会越过阈值，但会把负样本一起推到低 `mu`，导致全前景。
- bidirectional margin 明显更合理，能同时恢复非零 IoU 并避免全前景。
- 但 `bidirectional_margin_lambda1` 的 val/test IoU 仍低于 S85 `big_multichannel_v2`，说明 threshold crossing 是必要但不充分的修复。

## 下一步

- 以 bidirectional margin 为后续 margin objective 的基础。
- 继续调 `lambda_threshold_margin` / margins，并加入 validation-aware selection。
- 如果面积恢复但定位仍弱，下一步应加入 localization / boundary loss 或 direct mask + margin 组合。
