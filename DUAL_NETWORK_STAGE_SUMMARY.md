# DUAL_NETWORK_STAGE_SUMMARY

## S21 40x20 Adaptation Note

S21 reused the S20 `40x20` dataset and tested three BCE adaptation settings:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| S20 BCE reference | 1.739786e-01 | 2.125500e+02 | 2.053370e+05 | 3.794331e+02 |
| S21 bce_30steps_temp50 | 5.440697e-01 | 6.380000e+01 | 5.094976e+04 | 1.127006e+02 |
| S21 bce_30steps_temp25 | 9.262554e-01 | 3.290000e+01 | 3.503892e+04 | 1.461498e+02 |
| S21 bce_30steps_lambda3 | 9.053153e-01 | 3.280000e+01 | 1.327051e+04 | 6.555141e+01 |

S21 shows that the weaker S20 `40x20` result was strongly affected by resolution adaptation. Increasing training steps improves the result, lowering `mask_prior_temperature` to `25.0` gives the best average IoU, and increasing `lambda_mask_bce_prior` to `3.0` gives the best `mu_mse/mu_mae`.

This strengthens the semi-supervised dual-network route at higher resolution, but it does not change the core boundary: BCE mask prior uses `mu_label < 500`, so these are semi-supervised / diagnostic upper-bound results, not proof of unsupervised weak-form inversion success.

## S22 40x20 Combo Adaptation Note

S22 tested whether combining sharper mask temperature with stronger BCE weight can improve both IoU and `mu_mse/mu_mae`:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| S21 temp25 reference | 9.262554e-01 | 3.290000e+01 | 3.503892e+04 | 1.461498e+02 |
| S21 lambda3 reference | 9.053153e-01 | 3.280000e+01 | 1.327051e+04 | 6.555141e+01 |
| S22 combo_temp25_lambda3 | 9.177421e-01 | 3.260000e+01 | 3.814877e+04 | 1.629943e+02 |
| S22 combo_temp20_lambda3 | 9.172025e-01 | 3.265000e+01 | 5.109617e+04 | 1.986578e+02 |
| S22 combo_temp25_lambda5 | 9.139707e-01 | 3.230000e+01 | 4.181617e+04 | 1.753928e+02 |

Among S22 combinations, `combo_temp25_lambda3` is the most balanced candidate. However, it does not dominate S21: `bce_30steps_temp25` still has the best average IoU, and `bce_30steps_lambda3` still has the best `mu_mse/mu_mae`.

For 40x20 follow-up experiments, keep `bce_30steps_temp25` as the IoU-oriented default, keep `bce_30steps_lambda3` as the error-oriented default, and use `combo_temp25_lambda3` only when a single combined setting is required. This remains a semi-supervised / diagnostic upper-bound direction because BCE uses `mu_label < 500`; it is not evidence of pure unsupervised weak-form success.

## 1. 当前支线目标

本支线探索 `phi-Net / mu-Net` 双网络反演方法：

- `phi-Net` 输入坐标 `(x, y)`，输出磁标量势 `phi(x, y)`；
- `mu-Net` 输入坐标 `(x, y)`，输出磁导率分布 `mu(x, y)`；
- 使用变分场重构训练 `phi-Net`；
- 使用 `div(mu grad phi)=0` 的 weak-form material update 训练 `mu-Net`；
- 从单样本 prototype 推进到小规模 `.npz` runner；
- 当前支线不替代 `main`，也不声称优于 `main` 的 `signals / coords -> mu_map` 监督反演主线。

## 2. 已完成能力

- 模型和 loss 骨架已完成：`PhiNet`、`MuNet`、`energy_loss`、`data_loss`、`tv_loss`、`weak_form_loss`；
- `.npz` 数据接口已完成，支持 `coords` 或 `x/y` 两种坐标来源；
- compact-support weak-form test gradients 已实现：`generate_compact_support_test_grads`；
- single-sample loop 已跑通，能输出 `phi/mu` 闭环训练诊断；
- `minimal_dual_single_sample_loop.py` 已支持 area prior、Dice mask prior、BCE mask prior 和 diagnostics 输出；
- `train_dual_variational.py` 已从 skeleton 升级为小规模 runner，可对多个 sample 独立运行并输出 `metrics.csv`；
- S17/S18/S19 已完成 10/20/50-sample runner 验证；
- S20 已完成 40x20 网格、20-sample resolution probe。

## 3. 关键实验结论

### 无 BCE mask prior 时

在 S6-S13 中，`weak-form + TV + area / soft Dice` 能让训练 loss 下降，但材料反演质量不足：

- `defect_area_pred` 明显过大，常出现大面积低 `mu` 区域；
- IoU 长期偏低；
- 单靠固定 centers、signal-informed centers、label-informed centers 或更强 area prior 都不能稳定解决面积扩张；
- 这说明当前无监督 weak-form material update 的缺陷定位能力不足。

典型结果：

- S6 默认配置最终 `defect_area_pred=200`，`defect_area_label=11`，`defect_iou=0.055`；
- S7/S8 area prior 能压低预测面积，但 IoU 改善有限，较强 area prior 甚至出现面积更准但定位失败；
- S16 runner baseline 三样本平均 `defect_iou=9.287478e-02`，`defect_area_pred=64.0`；
- S17 runner baseline 十样本平均 `defect_iou=1.145468e-01`，`defect_area_pred=62.8`；
- S18 runner baseline 二十样本平均 `defect_iou=1.226982e-01`，`defect_area_pred=64.6`，`mu_mse=2.826850e+05`，`mu_mae=3.129512e+02`；
- S19 runner baseline 五十样本平均 `defect_iou=1.101394e-01`，`defect_area_pred=65.04`，`mu_mse=2.871279e+05`，`mu_mae=3.216540e+02`；
- S20 40x20 runner baseline 二十样本平均 `defect_iou=1.077682e-01`，`defect_area_pred=285.8`，`mu_mse=3.107496e+05`，`mu_mae=3.890810e+02`。

### label-informed centers

S11 使用 `mu_label` 的真实缺陷 centroid 构造 oracle centers：

- `label_three` / `label_nine` 能把预测 centroid 拉近 label centroid；
- IoU 相比固定 centers 有一定提升；
- 但 `defect_area_pred` 仍远大于 label area。

结论：

- center selection 是瓶颈之一；
- 但不是唯一瓶颈；
- 即使 centers 放在正确位置，当前 weak-form + area prior 仍倾向产生过宽的低 `mu` 区域。

### BCE mask prior

S14-S18 表明 BCE mask prior 是当前最有效的局部监督信号：

- S14 单样本中，`lambda_mask_bce_prior=1.0` 将 `defect_area_pred` 从 66 降到 11，`defect_iou` 从 0.166667 提升到 1.0；
- S15 三样本中，BCE 相比 baseline 稳定降低面积、提高 IoU、降低 `mu_mse/mu_mae`；
- S16 runner 复现 S15 结论；
- S17 十样本中，BCE 仍稳定优于 baseline；
- S18 二十样本中，BCE 继续稳定优于 baseline；
- S19 五十样本中，BCE 继续稳定优于 baseline；
- S20 40x20 网格中，BCE 平均优于 baseline，但改善明显弱于 S18/S19，且仍存在多样本低 IoU 和面积扩张。

S17 十样本平均：

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.145468e-01 | 6.280000e+01 | 2.766025e+05 | 3.173523e+02 |
| BCE | 8.425641e-01 | 7.800000e+00 | 1.319349e+04 | 6.156954e+01 |

S18 二十样本平均：

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.226982e-01 | 6.460000e+01 | 2.826850e+05 | 3.129512e+02 |
| BCE | 8.934921e-01 | 9.100000e+00 | 1.558897e+04 | 6.164804e+01 |

S19 五十样本平均：

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.101394e-01 | 6.504000e+01 | 2.871279e+05 | 3.216540e+02 |
| BCE | 8.399348e-01 | 9.120000e+00 | 1.935842e+04 | 6.434378e+01 |

S20 40x20 二十样本平均：

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.077682e-01 | 2.858000e+02 | 3.107496e+05 | 3.890810e+02 |
| BCE | 1.739786e-01 | 2.125500e+02 | 2.053370e+05 | 3.794331e+02 |

S20 说明 BCE 在更高分辨率下仍有半监督可行性迹象：平均 IoU、平均面积、`mu_mse` 和平均 `mu_mae` 优于 baseline。但该改善不如 S18/S19 稳定，BCE 只在 19/20 个样本上提高 IoU、在 9/20 个样本上降低 `mu_mae`，且多样本仍存在过大预测面积。因此 S20 不能作为高分辨率稳健性的充分证明。

但 BCE 使用 `mu_label < 500` 的真实 mask，因此它是半监督/诊断上界，不是无监督 weak-form 成功证明。

## 4. 当前最重要判断

当前支线已经证明：

1. 双网络 weak-form 框架可以工程上跑通；
2. 仅靠当前 weak-form + area/TV 约束，缺陷定位能力不足；
3. 加入 label mask 类局部监督后，结果显著改善；
4. S18/S19 的 20/50-sample 结果进一步说明 BCE mask prior 稳定优于 baseline；
5. S20 的 40x20 结果显示半监督路线在更高网格分辨率下仍有可行性迹象，但当前设置还不够分辨率稳健；
6. 因此下一阶段若继续，应转向半监督/弱监督方案的稳定化，或重新设计无监督物理约束，而不是继续简单扫 radius、centers、area prior。

## 5. 推荐下一阶段路线

推荐路线：半监督双网络支线。

理由：

- 当前实验已经证明 label mask prior 是最有效信号；
- S18/S19 已在 20/50 个样本上验证 BCE mask prior 稳定优于 baseline；
- S20 显示 BCE 在 40x20 网格下仍有平均改善，但需要更稳定的高分辨率训练设置；
- 继续纯无监督 weak-form 调参收益低；
- 半监督路线更容易形成可展示结果；
- 可以作为主线之外的结构探索，而不干扰 `main`。

建议下一阶段：

1. 固定 `train_dual_variational.py` runner；
2. 使用 BCE / Dice mask prior 作为可选半监督项；
3. 在 20-50 个小样本上继续测试，优先检查 S18 中的弱样本而不是继续扫 radius/centers；
4. 输出 `metrics.csv`；
5. 后续再考虑可视化和论文方法表述。

## 6. 当前不建议继续的方向

- 不建议继续单独扫 `test_radius`；
- 不建议继续扫 `center_mode`；
- 不建议继续强化 `area prior`；
- 不建议声称无监督 weak-form 反演成功；
- 不建议现在合并进 `main`。

## 7. 下一步最小执行建议

S19：semi-supervised runner consolidation / failure-sample review。

目标：

在 S18 已完成 20-sample 验证的基础上，不再继续单纯扫描 `test_radius`、`center_mode` 或 `area prior`，而是沿半监督双网络支线整理和验证：

- 复查 S18 中 IoU 相对较弱的样本；
- 继续比较 baseline: `weak-form + area + Dice` 与 semi-supervised: `baseline + BCE mask prior`；
- 必要时扩展到 20-50 个小样本；
- 记录 avg IoU、avg `defect_area_pred`、avg `mu_mse / mu_mae` 和 failure samples；
- 明确所有 BCE mask prior 结果都是半监督/诊断上界，不是无监督 weak-form 反演成功。
