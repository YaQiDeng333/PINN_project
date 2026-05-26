# S107 COMSOL V2 threshold-margin stage summary

## S103 诊断

- S101 三组都存在 `soft-hard mismatch`：soft foreground 面积非零，但 hard `defect_area_pred=0`。
- 三组最后的 `min_mu` 都高于 `500`，说明 area loss 没有推动 hard threshold crossing。
- `area_ratio_mse` 能降低连续 `mu_mse` / `mu_mae`，但没有让 `mu_pred` 跨过 `mu_threshold=500`。

## S105 结果

- `v2_baseline_reference` 仍为全背景，train / val / test IoU 均为 0。
- `positive_margin_lambda1` 和 `positive_margin_lambda10` 恢复 hard mask，但几乎全前景，`defect_area_pred=20000`。
- `bidirectional_margin_lambda1` 是 S105 最佳，train / val / test IoU = `1.305192e-01` / `1.185913e-01` / `1.266814e-01`，但仍明显低于 S85 `big_multichannel_v2`。

## 当前判断

- hard threshold crossing 是必要条件，但不是充分条件。
- positive-only margin 会把负样本也拉进缺陷区域，导致全前景。
- V2 下一步需要同时约束：
  - 正样本跨过 `mu_threshold`；
  - 负样本保持在背景侧；
  - 预测面积不要失控；
  - 最终 endpoint 不要因为最后若干 step 退化而错过较好状态。

## 下一步

- S108 增加 validation-aware best endpoint selection。
- S109 测试 bidirectional margin + area calibration。
- S109 同时测试 direct mask + area loss，判断 direct output path 是否比 hard `mu_threshold` 路径更适合 V2 multi-defect / small-label 任务。
