# S164 COMSOL center-aware full probe

## 执行原因

S163 `center_grid_loss` 同时满足 val/test gate：两者 mask IoU 均不低于同轮 1500-step reference，且至少一项提升超过 `0.03`，同时 center error 下降。因此执行 3000-step confirm。

## 配置

- output: `experiments/dual_network/S164_comsol_parametric_center_loss_full_probe/center_grid_loss_3000`
- `steps=3000`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- 其他配置沿用 raw MLP / shared head / fixed-order baseline。

## Metrics

| split | presence_acc | type_acc | continuous_mae | center_mae | center_grid_mae | center_axis_relative_mae | axis_mae | rotation_mae | depth_mae | mask_iou |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1.000000 | 1.000000 | 0.116999 | 0.000008 | 0.041154 | 0.002322 | 0.000032 | 0.701895 | 0.000018 | 0.726483 |
| val | 1.000000 | 0.616667 | 1.192975 | 0.001076 | 5.996350 | 0.325784 | 0.000613 | 7.154223 | 0.000247 | 0.469423 |
| test | 1.000000 | 0.633333 | 1.351877 | 0.001017 | 5.546025 | 0.314625 | 0.000615 | 8.107736 | 0.000260 | 0.498874 |

## 对比

- 相比 S115 / S158 pred_all historical baseline，val mask IoU 从 `0.369908` 提升到 `0.469423`，test 从 `0.424462` 提升到 `0.498874`。
- 相比 S163 1500-step `center_grid_loss`，val/test 保持相近改善；3000-step confirm 没有退化回原 baseline。
- 与 S117 / S158 oracle 仍有明显 gap，说明 center loss 是有效修复方向，但还没有解决全部 held-out localization。

## 自评

S164 full probe 支持继续 center-localization route。下一步不应简单加大 lambda，而应围绕更稳的 center representation 或 auxiliary center head 做结构性改进。
