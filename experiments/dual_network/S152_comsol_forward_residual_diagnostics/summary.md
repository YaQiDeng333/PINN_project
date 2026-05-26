# S152 COMSOL forward residual sensitivity diagnostic

## 目的

S152 检查 learned forward surrogate residual 是否能区分正确 / 错误 geometry。如果 residual 对 type / rotation / axis 错误不敏感，就不适合作为强 training loss。

## 配置

- split: val / test
- 数据: S84 COMSOL V2 converted NPZ
- targets: S113 raw parametric targets
- prediction source: S126 `s115_raw_mlp_export`
- variants:
  - `true_geometry`
  - `predicted_geometry`
  - `type_swapped_geometry`
  - `rotation_perturbed_geometry`，`+10 deg`
  - `axis_scaled_geometry`，`axis_x/y * 1.2`

## 结果

| split | variant | avg_signal_nrmse | avg_signal_corr |
| --- | --- | ---: | ---: |
| val | true_geometry | 2.680198e-01 | 9.572266e-01 |
| val | type_swapped_geometry | 7.348850e-01 | 7.968278e-01 |
| val | rotation_perturbed_geometry | 2.741137e+00 | 2.128940e-01 |
| val | axis_scaled_geometry | 2.661547e-01 | 9.577585e-01 |
| val | predicted_geometry | 7.572894e-01 | 7.595906e-01 |
| test | true_geometry | 2.606982e-01 | 9.590282e-01 |
| test | type_swapped_geometry | 4.385888e-01 | 8.902319e-01 |
| test | rotation_perturbed_geometry | 4.292691e+00 | 5.014552e-01 |
| test | axis_scaled_geometry | 2.691335e-01 | 9.568606e-01 |
| test | predicted_geometry | 7.149153e-01 | 7.132657e-01 |

Predicted geometry 的 residual 与 mask IoU 相关性：

- val: `-2.503890e-01`
- test: `-4.378152e-01`

## 结论

- Forward residual 对 `rotation_perturbed_geometry` 很敏感。
- Forward residual 对 `type_swapped_geometry` 中等敏感，val 更明显，test 较弱。
- Forward residual 对 `axis_scaled_geometry` 基本不敏感；val 中 axis_scaled residual 甚至略低于 true_geometry。
- Predicted geometry 的 residual 与 mask IoU 呈负相关，但相关性不强，不能单独作为 mask IoU proxy。

## 对 S149 的解释

S149 中 forward consistency loss 没有改善 mask IoU，可能因为 residual 主要惩罚 waveform-level rotation/type mismatch，但对 axis / mask-area 相关几何不敏感。作为训练 loss 时，它容易牺牲 mask IoU 来降低 signal residual。

## Review / 自评

- Claude Code review 已尝试两次：第一次因 budget limit 退出，第二次超时。
- 本地 smoke / py_compile 通过。
- 自检 `target_schema`、variant construction 和 train-only normalization 路径后，未发现必须修复项。
- 当前将 axis insensitivity 记录为诊断结论，而不是实现 bug。

## 下一步

Forward residual 暂时只保留为 diagnostic，不继续作为默认 loss。S153/S154 转向更直接的 type / rotation targeted supervision。
