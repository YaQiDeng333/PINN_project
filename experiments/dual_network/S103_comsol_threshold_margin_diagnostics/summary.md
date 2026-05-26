# S103 COMSOL V2 hard-threshold margin diagnostics

## 目的

S103 诊断 S101 中为什么 area loss 改善了 soft foreground / 连续 `mu`，但 hard mask 仍为全背景。重点检查 `soft_defect`、hard `defect_area_pred` 和 `mu_pred` 相对 `mu_threshold=500` 的 margin。

## 三组诊断结果

| run | hard_area_zero | soft_hard_mismatch | no_threshold_crossing | final_min_mu | final_mean_soft_defect | final_pred_area_soft_mean | final_true_area_mean |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| v2_baseline_with_history | True | True | True | 6.258309e+02 | 7.447070e-02 | 3.050320e+02 | 2.187300e+02 |
| area_ratio_mse | True | True | True | 5.413740e+02 | 8.250425e-02 | 3.379374e+02 | 2.264800e+02 |
| foreground_floor | True | True | True | 6.272391e+02 | 7.268819e-02 | 2.977308e+02 | 2.114400e+02 |

## 结论

- 三组都出现 hard `defect_area_pred=0`。
- 三组都出现 `soft_hard_mismatch`：soft foreground 面积非零，但 hard mask 仍为全背景。
- 三组最后一条 history 的 `min_mu` 都高于 `500`，说明没有任何 sampled point 的 `mu_pred` 跨过 hard threshold。
- `area_ratio_mse` 将 `min_mu` 从约 `626` 降到约 `541`，并降低了连续 `mu_mse`，但仍未跨过 `mu_threshold=500`。
- 当前问题不是 soft foreground 完全塌缩，而是 hard threshold crossing 不足。

## 建议

需要 S104 的 threshold-margin loss，直接推动正样本 `mu_pred < 500 - margin`，并测试是否能恢复非零 hard mask / IoU。
