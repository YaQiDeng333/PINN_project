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

## S23 Fresh 40x20 Candidate Validation Note

S23 generated a fresh `40x20` / 20-sample dataset and compared the two S21 candidate settings:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| temp25_lambda1 | 9.560439e-01 | 3.555000e+01 | 3.528870e+04 | 1.481365e+02 |
| temp50_lambda3 | 9.283914e-01 | 3.530000e+01 | 1.459990e+04 | 7.058641e+01 |

S23 reproduces the S21 pattern on new data: `temp25_lambda1` is the IoU-oriented 40x20 default candidate, while `temp50_lambda3` remains the continuous-`mu` error-oriented candidate. No obvious failure sample was observed.

This strengthens the 40x20 semi-supervised runner direction, but it still does not change the core boundary: BCE mask prior uses `mu_label < 500`, so these are semi-supervised / diagnostic upper-bound results, not evidence of pure unsupervised weak-form inversion success.

## S24 40x20 50-Sample Default Validation Note

S24 validated the S23 IoU-priority default candidate `temp25_lambda1` on a fresh `40x20` / 50-sample dataset:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.426606e-01 | 2.353600e+02 | 2.503184e+05 | 2.932973e+02 |
| temp25_lambda1 | 9.203000e-01 | 3.340000e+01 | 3.598542e+04 | 1.498937e+02 |

`temp25_lambda1` improves IoU on all 50 samples and is suitable as the current `40x20` IoU-priority default candidate. Three weak samples remain below IoU `0.7`, so failure-case review is still useful. S23's `temp50_lambda3` remains the continuous-`mu` error-oriented comparison.

This does not change the boundary: BCE mask prior uses `mu_label < 500`, so S24 is a semi-supervised / diagnostic upper-bound result, not proof of unsupervised weak-form inversion success.

## S25 80x40 High-Resolution Feasibility Note

S25 tested `temp25_lambda1` on a new `80x40` / 10-sample dataset with `20/20/20` training steps:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 6.824286e-02 | 1.418300e+03 | 3.996649e+05 | 4.761472e+02 |
| temp25_lambda1 | 5.102481e-01 | 2.611000e+02 | 1.519206e+05 | 3.708853e+02 |

`temp25_lambda1` improves IoU on all 10 samples and reduces predicted defect area, so the semi-supervised BCE upper-bound trend persists at `80x40`. However, absolute IoU is much weaker than S24 `40x20`, with weak samples at sample 6 and sample 9.

This indicates high-resolution feasibility, not high-resolution stability. If the branch continues at `80x40`, it needs resolution-specific adaptation such as more training steps, adjusted `test_radius`, center layout, mask temperature, BCE weight, or network capacity. The boundary remains unchanged: this is semi-supervised / diagnostic upper-bound evidence, not unsupervised weak-form success.

## S26 80x40 BCE Adaptation Note

S26 reused the S25 `80x40` / 10-sample dataset and tested three adaptation settings with `30/30/30` training steps:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| S25 temp25_lambda1 reference | 5.102481e-01 | 2.611000e+02 | 1.519206e+05 | 3.708853e+02 |
| temp25_lambda1_30steps | 8.081555e-01 | 1.308000e+02 | 4.524894e+04 | 1.601105e+02 |
| temp25_lambda3_30steps | 8.706159e-01 | 1.068000e+02 | 4.723388e+04 | 1.832961e+02 |
| temp20_lambda3_30steps | 8.866546e-01 | 1.077000e+02 | 5.950246e+04 | 2.175080e+02 |

S26 shows that the weaker S25 result was strongly affected by insufficient adaptation at `80x40`. More training steps plus sharper / stronger BCE settings substantially improve IoU and reduce over-expanded defect area. `temp20_lambda3_30steps` gives the best average IoU, while `temp25_lambda3_30steps` is the most balanced 80x40 follow-up candidate because it keeps near-best IoU with the lowest predicted area and lower continuous-`mu` errors than `temp20_lambda3_30steps`.

This does not change the core boundary: BCE mask prior uses `mu_label < 500`, so S26 remains a semi-supervised / diagnostic upper-bound result, not proof of unsupervised weak-form inversion success.

## S27 Fresh 80x40 Candidate Validation Note

S27 generated a fresh `80x40` / 20-sample dataset and compared `baseline`, `temp25_lambda3`, and `temp20_lambda3`:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.115430e-01 | 1.115400e+03 | 3.065218e+05 | 3.569112e+02 |
| temp25_lambda3 | 8.656310e-01 | 1.327500e+02 | 4.774737e+04 | 1.879750e+02 |
| temp20_lambda3 | 8.693352e-01 | 1.322500e+02 | 6.314803e+04 | 2.269721e+02 |

S27 validates that the S26 candidate configurations are not limited to the original S25 data. `temp20_lambda3` has the best average IoU and is the IoU-priority candidate. `temp25_lambda3` has much lower `mu_mse/mu_mae` with only slightly lower IoU, so it is the more balanced current `80x40` default candidate.

The boundary remains unchanged: BCE mask prior uses `mu_label < 500`, so S27 is semi-supervised / diagnostic upper-bound evidence, not proof of unsupervised weak-form inversion success.

## S28 80x40 50-Sample Default Validation Note

S28 generated a fresh `80x40` / 50-sample dataset and validated the S27 comprehensive default candidate `temp25_lambda3` against baseline:

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.051006e-01 | 1.242440e+03 | 3.443720e+05 | 3.814918e+02 |
| temp25_lambda3 | 8.925113e-01 | 1.328200e+02 | 4.572207e+04 | 1.832775e+02 |

`temp25_lambda3` improves IoU on all 50 samples, strongly reduces predicted defect area, and substantially lowers continuous `mu` errors. Two samples remain below IoU `0.7` and one is borderline, so failure-case review is still useful.

S28 supports `temp25_lambda3` as the current `80x40` comprehensive default candidate. The boundary remains unchanged: BCE mask prior uses `mu_label < 500`, so S28 is semi-supervised / diagnostic upper-bound evidence, not proof of unsupervised weak-form inversion success.

## S29 80x40 可视化失败诊断

S29 只读取已有 S28 输出，为 `temp25_lambda3` 整理代表性图像和失败样本表格。

- IoU 最高的成功样本：2、29、47。
- IoU 最低的弱样本：45、48、41、49、21。
- 面积误差最大的样本：21、45、3、37、11。
- centroid 偏移最大的样本：45、21、25、44、6。

S29 的诊断结论是：`temp25_lambda3` 已经基本修正 baseline 中全域扩张的低 `mu` 模式。剩余问题更可能来自形状细节不匹配、边界 / 窄缺陷样本、centroid 偏移或局部几何误差。下一步应转向最终结果整理和定向失败样本复查，而不是继续大范围参数扫描。

边界不变：`BCE mask prior` 使用 `mu_label < 500`，因此 S29 是半监督 / 诊断上界分析，不是无监督 weak-form 反演成功证明。

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

## 8. S30 跨分辨率结果报告

S30 已形成跨分辨率结果报告 `DUAL_NETWORK_RESULTS_REPORT.md`，并在 `experiments/dual_network/S30_cross_resolution_report/` 中保存 `aggregated_metrics.csv`、summary 和图表。

核心结论：BCE 半监督上界在 20x10、40x20、80x40 三个分辨率下均稳定优于 baseline。20x10 使用 `S19 bce`，40x20 使用 `temp25_lambda1`，80x40 使用 `temp25_lambda3`。该结果说明半监督双网络路线有稳定展示价值，但不能表述为纯无监督 weak-form 反演成功。

下一阶段建议从参数扫描转向结果整理、可视化失败样本分类和论文式支线表述。

## 9. S31 跨分辨率图表补齐

S31 使用纯 Python 标准库从 S30 `aggregated_metrics.csv` 生成了三张 SVG 图表：`defect_iou_by_resolution.svg`、`defect_area_pred_by_resolution.svg` 和 `mu_error_by_resolution.svg`。顶层 `DUAL_NETWORK_RESULTS_REPORT.md` 已增加图表索引。

当前支线结果展示材料更完整，但边界判断不变：BCE 是半监督 / 诊断上界，不是无监督 weak-form 成功。

S31 之后，支线已形成 README、术语、实验日志、阶段总结、结果报告和 artifact index 的文档体系。

S33 已补充复现实验说明 `DUAL_NETWORK_REPRODUCE.md`，支线文档体系包含 README、术语、实验日志、阶段总结、结果报告、成果索引和复现说明。

## 10. S34 100x50 中高分辨率验证

S34 使用新的 100x50 / 30-sample 数据验证 80x40 综合默认候选 `temp25_lambda3`。结果显示：baseline avg `defect_iou=1.036250e-01`，`temp25_lambda3` avg `defect_iou=8.559784e-01`；baseline avg `defect_area_pred=1.954133e+03`，`temp25_lambda3` avg `defect_area_pred=2.158667e+02`。

`temp25_lambda3` 在 30/30 个样本上 IoU 优于 baseline，说明该半监督候选在 100x50 下仍稳定有效，并为后续向 200x100 默认分辨率推进提供可行性迹象。边界不变：BCE 是半监督 / 诊断上界，不是无监督 weak-form 成功。

## 11. S35 200x100 默认分辨率 fullrun

S35 fullrun 复用上一轮已生成的 200x100 / 30-sample `.npz`，完整运行 baseline 与 `temp25_lambda3` 两组 30/30/30 runner，没有使用 partial 结果。

结果显示：baseline avg `defect_iou=7.936600e-02`，`temp25_lambda3` avg `defect_iou=4.285627e-01`；baseline avg `defect_area_pred=1.253147e+04`，`temp25_lambda3` avg `defect_area_pred=3.226300e+03`。

`temp25_lambda3` 在 30/30 个样本上 IoU 优于 baseline，说明半监督双网络路线具备默认分辨率可行性迹象。与此同时，200x100 的平均 IoU 低于 100x50，并存在弱样本，因此后续若继续默认分辨率，应关注网络容量、训练步数、loss 尺度和高分辨率 shape-detail / false-positive 控制。边界不变：BCE 是半监督 / 诊断上界，不是无监督 weak-form 成功。

## 12. S36 200x100 BCE 适配实验

S36 复用 S35 fullrun 的 200x100 / 30-sample 数据，测试 `outer_steps=60`、更强 BCE 权重和更锐利 mask temperature。S35 `temp25_lambda3` 参考结果为 avg `defect_iou=4.285627e-01`、avg `defect_area_pred=3.226300e+03`、avg `mu_mse=1.482883e+05`、avg `mu_mae=2.686302e+02`。

S36 结果显示：`temp25_lambda3_outer60` avg `defect_iou=6.510969e-01`，`temp25_lambda5_outer60` avg `defect_iou=7.331974e-01`，`temp20_lambda3_outer60` avg `defect_iou=6.404690e-01`。其中 `temp25_lambda5_outer60` 同时取得最低 avg `defect_area_pred=1.689667e+03`、avg `mu_mse=7.166056e+04` 和 avg `mu_mae=1.822885e+02`。

当前判断：200x100 下 S35 的弱表现主要受训练轮数和 BCE 权重影响；`temp25_lambda5_outer60` 是当前 200x100 后续候选。边界不变：BCE 是半监督 / 诊断上界，不是无监督 weak-form 成功。

## 13. S37 fresh 200x100 候选验证

S37 使用新的 200x100 / 30-sample 数据验证 S36 的两个候选：综合候选 `temp25_lambda5_outer60` 和 IoU 优先候选 `temp20_lambda3_outer60`，并加入 `baseline` 对照。

结果显示：baseline avg `defect_iou=6.917387e-02`，`temp25_lambda5_outer60` avg `defect_iou=8.093492e-01`，`temp20_lambda3_outer60` avg `defect_iou=6.485412e-01`。同时 `temp25_lambda5_outer60` 取得最低 avg `defect_area_pred=1.127800e+03`、avg `mu_mse=6.106250e+04` 和 avg `mu_mae=1.837940e+02`。

当前判断：S37 没有复现 `temp20_lambda3_outer60` 作为 200x100 IoU 最优候选；`temp25_lambda5_outer60` 在 IoU、面积控制和连续 `mu` 误差上都更稳，因此是当前 200x100 默认候选。边界不变：BCE 是半监督 / 诊断上界，不是无监督 weak-form 成功。
## 14. S38 200x100 / 50-sample 默认候选验证

S38 使用新的 200x100 / 50-sample 数据验证 S37 确定的默认候选 `temp25_lambda5_outer60`。结果显示：baseline avg `defect_iou=6.851125e-02`，`temp25_lambda5_outer60` avg `defect_iou=7.783885e-01`；baseline avg `defect_area_pred=1.121512e+04`，`temp25_lambda5_outer60` avg `defect_area_pred=1.197960e+03`；baseline avg `mu_mse=5.193801e+05` / `mu_mae=5.343024e+02`，`temp25_lambda5_outer60` avg `mu_mse=6.309463e+04` / `mu_mae=1.870116e+02`。

`temp25_lambda5_outer60` 在 50/50 个样本上 IoU 都优于 baseline，因此可以作为当前 200x100 半监督 runner 默认候选。仍存在弱样本，尤其是 sample 40、12、34、21、36，后续若继续推进应优先做弱样本可视化与失败类型分类。边界不变：BCE 是半监督 / 诊断上界，不是无监督 weak-form 成功。
## 15. S39 200x100 / 100-sample 默认候选验证

S39 使用新的 200x100 / 100-sample 数据验证 S38 默认候选 `temp25_lambda5_outer60`。结果显示：baseline avg `defect_iou=7.472148e-02`，`temp25_lambda5_outer60` avg `defect_iou=7.595984e-01`；baseline avg `defect_area_pred=1.095210e+04`，`temp25_lambda5_outer60` avg `defect_area_pred=1.333170e+03`；baseline avg `mu_mse=5.040013e+05` / `mu_mae=5.185692e+02`，`temp25_lambda5_outer60` avg `mu_mse=6.412482e+04` / `mu_mae=1.794309e+02`。

`temp25_lambda5_outer60` 在 100/100 个样本上 IoU 都优于 baseline，因此继续适合作为当前 200x100 半监督 runner 默认候选。仍存在弱样本，尤其是 sample 40、47、17、57、73、95、54、29、49、43、19、86；后续若继续推进，应优先做弱样本可视化与失败类型分类。边界不变：BCE 是半监督 / 诊断上界，不是无监督 weak-form 成功。


## 16. S40 200x100 failure diagnostics

S40 reads the completed S39 `baseline` and `temp25_lambda5_outer60` metrics, then generates a visual failure diagnostic report under `experiments/dual_network/S40_200x100_failure_diagnostics/` without running new training. The report includes the highest-IoU samples (5, 45, 81, 50, 77), lowest-IoU samples (40, 47, 17, 57, 73, 95, 54, 29, 49, 43), largest area-error samples, largest centroid-offset samples, and baseline-to-final improvement rankings.

The current 200x100 default candidate remains overall effective, but weak samples concentrate in false positives, area overprediction, centroid shift, and local shape-detail mismatch. The boundary remains unchanged: this is a semi-supervised BCE mask-prior upper bound, not unsupervised weak-form success.


## 17. S41 200x100 false-positive suppression probe

S41 reuses the S39 200x100 / 100-sample `.npz` and tests stronger false-positive suppression settings: `area3_bce5`, `area5_bce5`, and `area3_bce7`. Compared with the S39 default `temp25_lambda5_outer60` avg `defect_iou=7.595984e-01` and avg `defect_area_pred=1.333170e+03`, the strongest S41 result is `area3_bce7` with avg `defect_iou=8.209086e-01`, avg `defect_area_pred=1.074100e+03`, avg `mu_mse=5.008512e+04`, and avg `mu_mae=1.641459e+02`.

Current 200x100 follow-up candidate: `area3_bce7`. It improves IoU while reducing predicted defect area, so stronger BCE with moderate area prior is more useful than blindly increasing area-prior weight. The boundary remains unchanged: BCE is a semi-supervised / diagnostic upper bound, not unsupervised weak-form success.


## 18. S42 fresh 200x100 area3_bce7 validation

S42 generated a fresh 200x100 / 100-sample dataset and compared `baseline`, the previous default `temp25_lambda5_outer60`, and the S41 candidate `area3_bce7`. The average results were: baseline `defect_iou=7.703941e-02`, `defect_area_pred=1.146581e+04`; `temp25_lambda5_outer60` `defect_iou=7.398354e-01`, `defect_area_pred=1.429090e+03`; `area3_bce7` `defect_iou=8.047192e-01`, `defect_area_pred=1.213090e+03`.

Current 200x100 default candidate: `area3_bce7`. It outperforms `temp25_lambda5_outer60` on average IoU, predicted defect area, `mu_mse`, and `mu_mae` on fresh data, while remaining within the same semi-supervised BCE diagnostic-upper-bound setting. The boundary remains unchanged: this is not unsupervised weak-form success.


## 19. S43 200x100 area3_bce7 failure diagnostics

S43 reads the completed S42 `baseline`, `temp25_lambda5_outer60`, and `area3_bce7` metrics, then generates a failure diagnostic report under `experiments/dual_network/S43_200x100_area3_bce7_failure_diagnostics/` without running new training. The report includes the highest-IoU `area3_bce7` samples (47, 25, 85, 17, 15), lowest-IoU samples (28, 5, 71, 23, 11, 21, 8, 72, 99, 83, 82, 36, 27, 52, 7), largest area-error samples, largest centroid-offset samples, and comparison rankings against both baseline and `temp25_lambda5_outer60`.

Current 200x100 default candidate remains `area3_bce7`. The main residual problems concentrate in false positives, area overprediction, centroid shifts, and local shape-detail errors. The boundary remains unchanged: BCE is a semi-supervised / diagnostic upper bound, not unsupervised weak-form success.


## 20. S44 200x100 post-processing threshold diagnostics

S44 reads the completed S42 `area3_bce7` `final_mu_pred.npy` / `final_mu_label.npy` outputs and evaluates direct thresholding, largest connected component filtering, and oracle area top-k without retraining. The raw S42 mask `mu_pred < 500` has avg `defect_iou=8.047192e-01` and avg `defect_area_pred=1.213090e+03`. Direct threshold sweep does not beat threshold `500`; lower thresholds reduce false-positive area but reduce IoU as well.

Largest connected component filtering at threshold `500` gives only a small average improvement, avg `defect_iou=8.058842e-01` with avg `defect_area_pred=1.181090e+03`. Oracle area top-k is lower than raw thresholding, avg `defect_iou=7.788534e-01`, so the current bottleneck is not mainly simple threshold or area calibration; residual error is partly in the `mu_pred` spatial ranking / shape signal itself. The boundary remains unchanged: BCE is a semi-supervised / diagnostic upper bound, not unsupervised weak-form success.


## 21. S45 200x100 model capacity probe

S45 adds branch-local runner capacity arguments, `--hidden-dim` and `--num-layers`, with defaults kept at the existing 32x2 behavior. It then reuses the S42 200x100 data and compares `cap_32x2`, `cap_64x3`, and `cap_128x4` on samples 0-29 under the same `area3_bce7` loss/prior setup.

The best S45 result is `cap_128x4`: avg `defect_iou=8.680831e-01`, avg `defect_area_pred=1.179267e+03`, avg `mu_mse=4.888211e+04`, and avg `mu_mae=1.657460e+02`. `cap_32x2` remains the stable baseline with avg `defect_iou=7.581612e-01`, while `cap_64x3` underperforms on average. Current recommendation: treat `cap_128x4` as the next 200x100 capacity candidate to validate on fresh/larger data, but do not call it a final default until stability is confirmed. The boundary remains unchanged: BCE is a semi-supervised / diagnostic upper bound, not unsupervised weak-form success.


## 22. S46 fresh 200x100 capacity validation

S46 generated a fresh 200x100 / 50-sample dataset and compared `cap32_area3_bce7` against `cap128_area3_bce7`. The average results were: `cap32_area3_bce7` avg `defect_iou=8.252710e-01`, avg `defect_area_pred=1.071520e+03`, avg `mu_mse=5.076185e+04`, avg `mu_mae=1.663466e+02`; `cap128_area3_bce7` avg `defect_iou=8.628616e-01`, avg `defect_area_pred=8.390600e+02`, avg `mu_mse=4.923852e+04`, avg `mu_mae=1.764913e+02`.

Current 200x100 capacity candidate: `cap128_area3_bce7`, but with a stability caveat. It improves the average IoU and reduces severe failures, predicted area, and `mu_mse`, but it improves only 19/50 samples and regresses on 31/50 samples, with worse average `mu_mae`. S46 therefore supports `128x4` as the next capacity candidate for larger validation, not as an unconditional final default. The boundary remains unchanged: BCE is a semi-supervised / diagnostic upper bound, not unsupervised weak-form success.


## 23. S47 200x100 100-sample capacity validation

S47 generated a fresh 200x100 / 100-sample dataset and compared `cap32_area3_bce7` against `cap128_area3_bce7`. The average results were: `cap32_area3_bce7` avg `defect_iou=8.341983e-01`, avg `defect_area_pred=9.976200e+02`, avg `mu_mse=5.061777e+04`, avg `mu_mae=1.742169e+02`; `cap128_area3_bce7` avg `defect_iou=8.475844e-01`, avg `defect_area_pred=9.081600e+02`, avg `mu_mse=5.336808e+04`, avg `mu_mae=1.851059e+02`.

Current 200x100 capacity default should remain `32x2 + area3_bce7`. `cap128_area3_bce7` has better average IoU and lower predicted area, but it improves only 45/100 samples, regresses on 55/100 samples, and has worse average `mu_mse` / `mu_mae`. S47 therefore downgrades `128x4` from a possible default replacement to a high-capacity diagnostic candidate. The boundary remains unchanged: BCE is a semi-supervised / diagnostic upper bound, not unsupervised weak-form success.


## 24. S48 signal-conditioned dual-network skeleton

S48 marks the transition from the per-sample semi-supervised optimization stage to the signal-conditioned model stage. The per-sample runner stage has established that the weak-form dual-network path is engineerable and that `BCE mask prior` gives a stable diagnostic upper bound across the tested resolutions, but it still trains a separate `PhiNet` / `MuNet` per sample and uses `mu_label` / `label_mask` in the strongest results.

S48 adds `CONDITIONAL_DUAL_NETWORK_PLAN.md`, `conditional_dual_models.py`, `smoke_test_conditional_dual_models.py`, and `experiments/dual_network/S48_conditional_model_skeleton/summary.md`. The new `ConditionalDualNet` uses `BzEncoder` to encode `signals`, then predicts `mu` and `phi` from `coords + latent`. The intended inference boundary is now explicit: a deployable branch candidate must use only `Bz signal + coords` at inference time and must not depend on `mu_label` or `label_mask`.

S48 does not train a model and has no performance metrics. The next practical step is a conditional runner data interface / batch loader so S49 can test a tiny supervised batch loop before any larger comparison with the main baseline.


## 25. S49 conditional dual-network data interface

S49 completes the first conditional model data-interface step. The conditional stage can now load `.npz` files containing `signals`, `mu_maps`, and either `coords` or `x/y`, then build batch tensors for `ConditionalDualNet`: `signals [B,signal_len]`, `coords [N,2]`, `mu_label [B,N,1]`, and `mask_label [B,N,1]`.

S49 also applies the S48 review cleanup: Xavier initialization for conditional model `Linear` layers and a backward smoke test for the model skeleton. Both the model smoke test and the data utility smoke test pass. No formal training was run, no checkpoint was saved, and the branch still has no conditional-model performance metrics.

The next stage should be S50: a conditional supervised training runner skeleton with a tiny batch smoke-train loop. The inference boundary remains unchanged: future deployable comparisons must use only `signals + coords`, not `mu_label` or `label_mask`.


## 26. S50 conditional supervised training runner skeleton

S50 creates `train_conditional_dual.py`, the first minimal supervised runner for `ConditionalDualNet`. It reads a conditional batch from `.npz`, trains a shared signal-conditioned model using mask BCE, mask Dice, and optional `mu_mse`, then writes `metrics.csv` and `run_summary.md` without saving model weights, checkpoints, arrays, or figures.

This is the first executable step from per-sample optimization toward a reusable conditional model. The runner still uses `mu_label` / `mask_label` during training, but the model forward path is `signals + coords -> mu / phi`. S50 does not include weak-form / physics loss and does not provide formal experiment metrics.

The branch still has not been compared against the main baseline in the conditional-model setting. The next step should be S51: run the supervised conditional runner on a real small `.npz` and confirm it can learn train-sample masks before adding validation or weak-form losses.


## 27. S51 conditional supervised small-data probe

S51 moves the conditional model stage from skeleton into the first real small-data train-set probe. It generates a fresh 20x10 / 20-sample `.npz` dataset and runs `train_conditional_dual.py` for 300 supervised steps on all 20 train samples using mask BCE + Dice loss.

The final train averages are `defect_iou=5.228869e-01`, `defect_area_pred=5.050000e+00`, `mu_mse=3.178682e+04`, and `mu_mae=1.286587e+02`, with all metrics finite. This shows the conditional supervised runner can learn nontrivial train-sample mask signal from real generated data, but the result is not yet strong and includes failed train samples.

Boundary unchanged: S51 is only a train-set probe, not a test-set generalization result and not a main-baseline comparison. The next step should establish train/val/test conditional evaluation before adding weak-form loss or making broader claims.


## 28. S52 conditional train-set overfit probe

S52 reuses the S51 20x10 train data and tests whether the conditional supervised model can fit the train set more strongly. Increasing training to 1000 steps raises average train IoU from S51 `5.228869e-01` to `8.211111e-01`, showing S51 was partly undertrained. Adding light `mu_mse` improves continuous errors but slightly lowers IoU in this run. Increasing capacity to `hidden_dim=128`, `num_layers=4`, `latent_dim=64` gives the best train result: avg `defect_iou=9.375000e-01`, avg `defect_area_pred=5.950000e+00`, avg `mu_mse=2.941439e+03`, and avg `mu_mae=1.233218e+01`.

Current conditional train-set fitting ability is therefore much stronger than S51 suggested, and the result is capacity-sensitive. The next step should be a train/val/test conditional runner before making any generalization or main-baseline comparison. Boundary unchanged: S52 is still a train-set overfit probe, not test-set performance.


## 29. S53 conditional train/val generalization probe

S53 extends `train_conditional_dual.py` with optional eval `.npz` support and starts validating conditional train/val behavior on 20x10 data. The `big_bce_dice` configuration strongly fits 80 train samples, with train avg `defect_iou=9.350000e-01`, but held-out val avg `defect_iou` is only `7.141148e-02`. This creates a large train-val IoU gap and indicates overfitting rather than useful conditional generalization.

The `big_bce_dice_mu1e-4` variant improves continuous val `mu_mse` / `mu_mae` relative to `big_bce_dice`, but collapses predicted masks to empty outputs on both train and val, giving avg `defect_iou=0.000000e+00`.

S53 therefore does not support main-baseline comparison yet. It shows the conditional supervised runner can fit train data but needs validation-aware model selection, regularization / capacity control, signal encoder improvements, or loss balancing before weak-form loss or larger claims are justified.


## 30. S54 conditional train/val/test generalization probe

S54 extends `train_conditional_dual.py` with optional test `.npz` support. The runner now keeps `metrics.csv` for train metrics, writes `eval_metrics.csv` when val/eval data is provided, writes `test_metrics.csv` when test data is provided, and reports all available split averages in `run_summary.md`.

On a fresh 20x10 dataset with 200 train, 50 val, and 50 test samples, `medium_bce_dice` reaches train avg `defect_iou=8.627310e-01`, but val/test IoU are only `5.660991e-02` / `8.526794e-02`. `big_bce_dice` reaches train avg `defect_iou=9.397538e-01`, with val/test IoU `7.397455e-02` / `7.558780e-02`.

S54 therefore confirms the conditional runner can fit train samples but still does not generalize well to held-out val/test samples. The result only represents a 20x10 small-scale supervised conditional probe, not a final replacement for the main baseline. Next work should focus on conditional generalization and loss/encoder design before adding weak-form loss or moving to larger resolutions.


## 31. S55 conditional data-scale generalization probe

S55 scales the 20x10 conditional supervised dataset to 1000 train, 200 val, and 200 test samples to check whether S54's weak held-out IoU was mainly caused by insufficient train data. The medium model reaches train / val / test avg `defect_iou=6.568019e-01` / `9.390210e-02` / `8.759288e-02`. The big model reaches train / val / test avg `defect_iou=9.130737e-01` / `8.848209e-02` / `7.708859e-02`.

Compared with S54, more data gives a modest val/test improvement, but held-out IoU remains low. The big model still overfits strongly; the medium model has slightly better held-out IoU and smaller train-to-held-out gaps. S55 therefore indicates that 20x10 conditional generalization is not solved by sample count alone. The next step should stay on conditional generalization diagnostics, regularization, signal encoder design, augmentation, and loss balancing. The branch boundary is unchanged: this is supervised conditional training, not unsupervised weak-form success and not a main-baseline replacement.


## 32. S56 conditional signal ablation probe

S56 adds `--signal-ablation` to `train_conditional_dual.py` and evaluates the trained conditional model with correct signals, zero signals, and shuffled signals. It reuses the S55 20x10 data and the big BCE + Dice configuration.

Correct signal IoU is train / val / test `9.072941e-01` / `9.372990e-02` / `8.830051e-02`. Zero-signal IoU is `1.577167e-02` / `1.992248e-02` / `2.013694e-02`. Shuffled-signal IoU is `6.966472e-02` / `6.911540e-02` / `5.805073e-02`.

The model clearly uses `Bz signal` on train samples, and correct signals still beat zero/shuffled signals on held-out splits. However, val/test IoU remains low and the correct-vs-shuffled margin is small, so the current failure is not just data quantity and not purely coordinate-only behavior. The likely bottleneck is weak signal conditioning: encoder design, signal normalization, conditioning injection, regularization, or supervised loss design need attention before weak-form loss, higher resolution, or main-baseline comparison.


## 33. S57 conditional signal normalization probe

S57 adds `--signal-normalization` to `train_conditional_dual.py` and compares `none`, `train_zscore`, and `per_sample_zscore` on the S55 20x10 data with the big BCE + Dice configuration.

The three train / val / test IoU results are: `none` = `9.114212e-01` / `9.318296e-02` / `8.784143e-02`; `train_zscore` = `8.914478e-01` / `7.994475e-02` / `8.733596e-02`; `per_sample_zscore` = `9.116529e-01` / `9.567763e-02` / `9.598926e-02`.

`per_sample_zscore` is the best S57 setting on held-out IoU, but the improvement is modest and val/test IoU remains low. `train_zscore` does not help. S57 therefore suggests raw signal scale is not the main issue; the remaining bottleneck is more likely `BzEncoder` / conditioning architecture, regularization, and loss design. The branch is still not ready for main-baseline comparison.


## 34. S58 conditional FiLM conditioning probe

S58 adds `conditioning_mode` support to the conditional model stack and compares concat conditioning against FiLM-style latent modulation. The default remains `concat`; `film` uses latent-derived gamma / beta to modulate hidden coordinate features.

On the S55 20x10 data, `concat_per_sample_zscore_reference` reaches train / val / test IoU `8.649705e-01` / `7.280748e-02` / `1.063626e-01`. `film_per_sample_zscore` reaches `9.563841e-01` / `6.837561e-02` / `7.330513e-02`. `film_train_zscore` reaches `9.696210e-01` / `8.486297e-02` / `8.279666e-02`.

FiLM improves train fitting but does not improve held-out IoU; it increases the train-to-held-out gap in this setup. The best S58 test IoU remains the concat reference, while `film_train_zscore` has the best val IoU and lower continuous held-out errors. S58 therefore does not support changing the default conditioning mode to FiLM. The next conditional work should focus on encoder structure, regularization, validation-aware training, and loss design.


## 35. S59 conditional signal encoder architecture probe

S59 adds `encoder_type` support to the conditional model stack and compares the existing MLP `BzEncoder` with a new 1D CNN signal encoder. The default remains `encoder_type=mlp`; the new `cnn` option can also be combined with `conditioning_mode=concat` or `conditioning_mode=film`.

On the S55 20x10 data, `mlp_concat_reference` reaches train / val / test IoU `8.432535e-01` / `9.821131e-02` / `1.118232e-01`. `cnn_concat` reaches `9.442416e-01` / `9.915262e-02` / `9.681660e-02`. `cnn_film` reaches `9.844595e-01` / `1.254790e-01` / `1.094247e-01`.

The CNN encoder improves train fitting, and `cnn_film` gives the best S59 val IoU, but held-out gains are modest and not consistent on test. The train-to-held-out gaps remain large, so S59 does not support treating encoder architecture alone as the fix. The current conditional branch should keep `mlp_concat_reference` as the generalization baseline and use `cnn_film` as a high-capacity diagnostic configuration while focusing next on validation-aware training, regularization, supervised loss design, and dataset ambiguity.


## 36. S60 conditional local signal feature probe

S60 adds optional `point_features [B,N,K]` support to the conditional model stack and introduces `--point-signal-mode none|local_value|local_value_abs` in `train_conditional_dual.py`. This tests whether giving each coordinate point an x-aligned local Bz signal feature improves held-out generalization beyond the global latent vector.

On the S55 20x10 data, `no_local_signal_reference` reaches train / val / test IoU `9.165308e-01` / `8.535813e-02` / `1.000321e-01`. `local_value` reaches `8.968483e-01` / `8.595360e-02` / `9.525210e-02`. `local_value_abs` reaches `8.831255e-01` / `9.574179e-02` / `9.469099e-02`.

The local Bz features do not produce a stable held-out IoU improvement. `local_value_abs` gives the best S60 val IoU and lower test continuous errors, but its test IoU is below the no-local reference. S60 therefore suggests the conditional generalization bottleneck is not simply global latent compression of local signal information. The next conditional work should focus on validation-aware training, regularization, target / loss design, and dataset ambiguity.


## 37. S61 conditional direct mask head probe

S61 adds an optional direct mask head to `ConditionalDualNet`. With `predict_mask=True`, the model returns `mask_logits` and `mask_prob`; `train_conditional_dual.py` supports `--mask-head-mode mu_threshold|direct`. The default remains `mu_threshold`.

On the S55 20x10 data, `mu_threshold_reference` reaches train / val / test IoU `8.861030e-01` / `8.633440e-02` / `1.256267e-01`. `direct_mask_head` reaches `9.953174e-01` / `9.028429e-02` / `8.906158e-02`.

The direct mask head nearly saturates train IoU but does not improve held-out test IoU and has a larger train-to-held-out gap. Because `lambda_mu_mse=0.0`, the direct head also leaves `mu` effectively unoptimized, producing much worse continuous `mu_mse` / `mu_mae`. S61 therefore does not support replacing the current conditional baseline with direct mask prediction as-is. The likely bottleneck remains signal-to-shape generalization, loss balancing, validation-aware training, or dataset ambiguity.


## 38. S62 conditional direct mask multi-task loss probe

S62 reuses the S55 20x10 train / val / test data and tests whether adding a light `mu_mse` term to the S61 direct mask head can recover meaningful continuous `mu` predictions while preserving mask IoU. All three S62 runs use `mask_head_mode=direct`, `signal_normalization=per_sample_zscore`, and BCE + Dice mask supervision.

`direct_mu0_reference` reaches train / val / test IoU `9.828120e-01` / `1.051470e-01` / `9.803234e-02`, but keeps very large test `mu_mse=2.335287e+05` and `mu_mae=4.689150e+02`. Adding `lambda_mu_mse=1e-5` keeps near-perfect train IoU, with train / val / test IoU `9.937419e-01` / `9.816303e-02` / `9.670001e-02`, and reduces test `mu_mse` / `mu_mae` to `5.399731e+04` / `5.539699e+01`. Increasing to `lambda_mu_mse=1e-4` further disciplines continuous `mu` but lowers mask IoU, reaching train / val / test IoU `9.199159e-01` / `1.003297e-01` / `8.550005e-02`.

S62 shows that a light `mu_mse` term fixes the direct head's continuous `mu` error scale, but it does not improve held-out test IoU over the S61 `mu_threshold_reference`. `direct_mu1e-5` is the best direct-head multi-task diagnostic setting, while the conditional held-out baseline should remain `mu_threshold_reference`. The next work should prioritize validation-aware selection and signal-to-shape generalization rather than optimizing train IoU alone.


## 39. S63 conditional derived Bz signal feature probe

S63 adds `--signal-feature-mode raw|raw_abs_grad` to `train_conditional_dual.py`. The new `raw_abs_grad` mode constructs encoder inputs from the normalized raw Bz signal, `abs(Bz)`, and finite-difference Bz gradients, expanding the encoder input length from 20 to 60 on the current 20x10 S55 data. This is only a derived-feature test on the existing single Bz signal, not a COMSOL multi-height data experiment.

On the S55 20x10 data, `raw_reference` reaches train / val / test IoU `8.758866e-01` / `9.413832e-02` / `1.040388e-01`. `raw_abs_grad` reaches `9.142336e-01` / `1.081610e-01` / `1.036962e-01`.

The derived `raw_abs_grad` features improve train IoU and val IoU, but do not improve held-out test IoU. S63 therefore does not show a stable generalization gain from adding simple derived channels to the current single Bz input. If the conditional branch continues, the next stage should consider multi-height / COMSOL Bz data interfaces or clearer signal-to-shape dataset diagnostics rather than only expanding single-signal features.


## 40. S64 multi-height Bz signal interface skeleton

S64 starts the multi-channel / multi-height Bz input-interface stage. `conditional_dual_data_utils.py` now supports both `signals [num_samples, signal_len]` and `signals [num_samples, num_channels, signal_len]`. The 3D form is flattened channels-first into `[B, C*L]`, so the current `BzEncoder` can keep receiving a flat `[B, signal_len]` tensor while the batch records channel metadata.

The updated batch includes `signal_original_shape`, `signal_channels`, `signal_length_per_channel`, `flattened_signal_length`, and `signal_flatten_order`. `train_conditional_dual.py` records the original signal shape and flattened signal length in `run_summary.md`. The data utility smoke test now checks `[4,3,20] -> [3,60]`, and the training smoke test runs a tempfile multi-channel train / eval / test pass.

S64 does not run formal training and does not generate COMSOL data. It is only an interface skeleton preparing for future synthetic multi-channel proxy probes or a COMSOL multi-height Bz `.npz` conversion path.


## 41. S65 synthetic multi-height Bz proxy probe

S65 adds `build_multiheight_proxy_npz.py` and creates a synthetic three-channel proxy from the existing S55 single Bz signals. The proxy channels are raw Bz, `window=3` smoothing times `0.8`, and `window=7` smoothing times `0.6`. This is not real COMSOL multi-height data; it is only a proxy to exercise the S64 multi-channel input path.

On the S55 20x10 data, `single_channel_reference` reaches train / val / test IoU `9.032763e-01` / `9.708926e-02` / `9.556112e-02`. `synthetic_multiheight_proxy` reaches `5.832855e-01` / `1.055699e-01` / `1.116188e-01`.

S65 confirms the multi-channel conditional runner can train and evaluate `[N,3,L]` proxy signals. The proxy improves held-out val/test IoU and continuous held-out `mu` errors relative to the S65 single-channel reference, but it has much lower train IoU and an unstable final endpoint. The result should be treated as an interface/proxy diagnostic, not as evidence that real multi-height physics is solved. The next meaningful step is a real COMSOL / multi-height Bz data path or a more physically grounded forward-data conversion.


## 42. S66 COMSOL-style multi-height Bz dataset interface

S66 starts the real COMSOL / multi-height Bz data-interface stage. It adds `COMSOL_MULTIHEIGHT_BZ_DATA_PLAN.md`, `comsol_multiheight_npz_utils.py`, and `smoke_test_comsol_multiheight_npz_utils.py`.

The recommended schema requires `signals`, `mu_maps` or `masks`, and `x/y` or `coords`. For COMSOL-style multi-height data, `signals` must be `[num_samples, num_channels, signal_len]` with at least two channels. Recommended metadata includes `signal_channel_names`, `lift_off_values`, `field_components`, `probe_line_y_values`, `geometry_units`, `field_units`, and `signal_flatten_order`.

The S66 smoke test uses a mock COMSOL-style file with `signals [5,3,20]`, validates the schema, loads it through the existing conditional data utilities, checks flattening to `[3,60]`, and runs a `ConditionalDualNet(signal_len=60)` forward pass. S66 does not call COMSOL, does not generate real COMSOL data, and does not run formal training.


## 43. S67 COMSOL multi-height CSV to NPZ converter

S67 增加 `convert_comsol_multiheight_csv_to_npz.py` 和 `smoke_test_convert_comsol_multiheight_csv_to_npz.py`，作为从 COMSOL-style long CSV signals 转换到 S66 multi-channel `.npz` schema 的第一版 converter entrypoint。

converter 需要 long-table signal columns：`sample_index`、`channel_index`、`channel_name`、`lift_off`、`field_component`、`x_index`、`x`、`value`。它会把 signal table 与包含 `mu_maps` 或 `masks`、以及 `x/y` 或 `coords` 的 target `.npz` 合并，输出 `signals [N,C,L]`，并写入 `signal_channel_names`、`lift_off_values`、`field_components`、`source_type`、`signal_flatten_order` 等 metadata。

S67 smoke test 只使用 mock CSV data：它用 `comsol_multiheight_npz_utils.py` 验证转换后的 `.npz`，检查 conditional data utility 是否按 `[B,C*L]` flatten，并运行一次 `ConditionalDualNet(signal_len=60)` forward pass。S67 不调用 COMSOL，不包含真实 COMSOL 数据，也不运行正式训练。
## 44. S68 COMSOL pilot data handoff

S68 将支线从接口 skeleton 推进到真实 COMSOL pilot 数据准备阶段。新增 `COMSOL_PILOT_DATA_REQUEST.md`，明确 COMSOL 侧应生成的文件、long CSV schema、target NPZ schema、pilot 数据规模和回到本支线后的转换 / 验证命令。

第一批 pilot 建议只生成 5 到 10 个 samples，使用 grid_x = 200、grid_y = 100、probe x points = 200、3 个 Bz channels，以及 `lift_off_values = [0.5, 1.0, 2.0]`。

S68 当前没有真实 COMSOL 数据，也没有训练结果。下一步需要在 COMSOL MCP 项目或相关对话中生成真实 pilot，再回到本支线用 S67 converter 和 S66 validator 接入。
## 45. S69 COMSOL pilot handoff end-to-end dry-run

S69 新增 `smoke_test_comsol_pilot_handoff_end_to_end.py`，用 tempfile 模拟 COMSOL 侧的 `signals_multiheight.csv` 和 `targets.npz`，然后调用 S67 converter、S66 validator、conditional data utils 和 `ConditionalDualNet` forward。

dry-run 已覆盖 CSV -> NPZ -> validator -> conditional batch -> model forward 链路，说明真实 pilot 数据回来后，本支线具备先转换、验证、再进入训练前检查的基本路径。

S69 仍然不是真实 COMSOL 数据，不调用 COMSOL，也不运行正式训练。
## 46. S70 COMSOL MCP pilot prompt package

S70 新增 `COMSOL_MCP_PILOT_PROMPT.md`，把真实 COMSOL pilot 数据生成任务整理成可直接交给 COMSOL MCP / COMSOL 相关对话的 prompt。

到 S70 为止，本支线已经具备真实 COMSOL pilot 数据接入前的文档、converter、validator 和 end-to-end dry-run 链路。当前仍没有真实 COMSOL 数据，也没有模型训练结论。

下一步应切到 COMSOL MCP 项目生成 5-10 个真实 pilot samples，再回到本支线转换、验证和决定是否训练。

## 47. S71 real COMSOL pilot multi-height Bz ingest

S71 已完成第一批真实 COMSOL pilot 数据接入。原始 `signals_multiheight.csv`、`targets.npz` 和 `README.md` 被复制到 `experiments/dual_network/S71_comsol_pilot_ingest/raw/`，并通过 S67 converter 转换为 `converted/comsol_multiheight_pilot.npz`。

converted NPZ 的 `signals shape = [5,3,200]`，包含 `mu_maps` 和 `masks`。S66 validator 通过，conditional data utils 可将 batch flatten 为 `[B,600]`，`ConditionalDualNet(signal_len=600)` forward 也通过。

当前 pilot 固定仿体，只改动磁性参数，样本数也只有 5。因此 S71 只说明真实 COMSOL multi-height Bz 数据可以进入支线接口，不代表 conditional model 已具备形状泛化能力。

## 48. S72 real COMSOL pilot conditional sanity probe

S72 使用 S71 converted NPZ 运行了一个 5-sample、300-step 的 conditional supervised sanity probe。`train_conditional_dual.py` 成功读取 `signals [5,3,200]`，按 channels-first flatten 为 600 维 encoder input，并输出 `metrics.csv` / `run_summary.md`。

S72 final average metrics 为：`defect_iou = 4.767923e-01`，`defect_area_pred = 4.019000e+03`，`mu_mse = 9.244401e+04`，`mu_mae = 2.460563e+02`。所有 metrics 均为 finite。

该结果只证明真实 COMSOL pilot 可以进入 conditional runner；由于样本数少且固定仿体，不能用于判断 shape generalization。

## 49. S73 COMSOL geometry-variation data request

S73 明确下一批 COMSOL 数据必须变化缺陷几何。第一批 pilot 固定仿体、只改磁性参数，因此只能验证接口和 runner 链路；要判断 conditional model 是否能替代或补充当前支线，需要 train / val / test split 中包含 defect center、size、depth、shape 和 permeability / mu 的变化。

建议下一批优先做 train 50-100、val 10-20、test 10-20；如果 COMSOL 成本较高，可先做 train 20、val 5、test 5。该阶段是 conditional model 替代主线前的关键数据准备阶段。


## 50. S74/S75/S76 COMSOL geometry-variation ingest and first probe

S74 已完成第一批真实 COMSOL geometry-variation train / val / test 数据接入。S67 converter 将 long CSV + target NPZ 转成支线可读 multi-channel NPZ，train / val / test signals shape 分别为 `[50,3,200]`、`[10,3,200]`、`[10,3,200]`，conditional loader 可 flatten 为 `[B,600]`。

S75 完成第一轮真实 COMSOL multi-height conditional train / val / test probe。`medium_multichannel` 的 train / val / test IoU 为 `5.225618e-01` / `4.088045e-01` / `3.961416e-01`；`big_multichannel` 为 `5.391816e-01` / `4.067505e-01` / `3.997817e-01`。这明显高于早期 synthetic single-Bz conditional 阶段约 0.1 左右的 held-out IoU，说明真实 multi-height Bz 数据更有潜力。

S76 的当前判断是：结果仍属于 pilot 阶段，样本量小，`defect_type` 固定为 `ellipsoid`，未变化旋转角或边界不规则度。下一步应扩大 COMSOL geometry 数据规模，并检查 target/mask 定义、loss balance、validation-aware selection 和更丰富几何分布。


## 51. S77-S80 COMSOL target/mask and train-fit diagnostics

S77 检查了真实 COMSOL geometry 数据中的 `mu_maps` 与 provided `masks`：train / val / test 的 `mu_maps < 500` 与 `masks > 0.5` 完全一致，avg mask IoU 均为 1.0，mismatch count 均为 0。因此 S75 train IoU 偏低不是由 mask label 定义差异导致。

S78 增加 `mask_source=mu_threshold|masks` 并比较两条训练路径。`mu_threshold_reference` 的 train / val / test IoU 为 `5.401838e-01` / `4.041796e-01` / `4.047063e-01`；`provided_masks` 为 `5.403162e-01` / `4.011194e-01` / `3.888755e-01`。provided masks 没有改善，后续仍可默认使用 `mask_source=mu_threshold`。

S79 因 S78 最佳 train IoU 低于 0.70 而执行。`longer_steps`、`bigger_subsample`、`bce2_dice1` 均未把 train IoU 提升到 0.70，也没有带来稳定 held-out 改善。当前判断是：简单加 steps、增大 point subsample 或提高 BCE 权重不是主要解决方案；下一步应转向 target / loss / model 表达和数据分布诊断，同时扩大 COMSOL geometry 数据多样性。


## 52. S81-S83 COMSOL output head diagnostics and geometry V2 request

S81 ??? COMSOL geometry-variation multi-height Bz ?????? `mu_threshold_reference`?`direct_mu0` ? `direct_mu1e-5`?direct head ?????? IoU ???`direct_mu1e-5` ?????? `mu_mse` / `mu_mae`???????? held-out IoU??? `mu_threshold` ?? mask ????????????? baseline ?????

S82 ?? S74-S81 ???????????? data size/diversity???? model/conditioning???? loss/output head???? train fitting???? target/mask???????????? output head ??? loss ???

S83 ?? V2 COMSOL geometry-variation ??????????? train 200 / val 50 / test 50???? fallback ? train 100 / val 20 / test 20?V2 ???? defect type?rotation angle ? boundary irregularity??????? `defect_params` metadata?

## 53. S84-S86 COMSOL geometry V2 ingest and first probe

S84 已完成真实 COMSOL geometry V2 fallback 数据接入。`comsol_geometry_variation_v2_exports/` 中的 train / val / test long CSV + target NPZ 已复制到 S84 raw 目录，并通过 S67 converter 转成支线可读 multi-channel NPZ。converted train / val / test signals shape 分别为 `[100,3,200]`、`[20,3,200]`、`[20,3,200]`，conditional loader 可 flatten 为 `[B,600]`，validator 和 `ConditionalDualNet(signal_len=600)` forward 检查均通过。

S85 完成第一轮 V2 conditional train / val / test probe。`medium_multichannel_v2` 的 train / val / test IoU 为 `2.307939e-01` / `2.107340e-01` / `2.048062e-01`；`big_multichannel_v2` 为 `3.023806e-01` / `2.593440e-01` / `2.768323e-01`。`big_multichannel_v2` 是当前较优配置，但仍明显低于 V1 S75 的 held-out IoU 约 `0.40`。

S86 的判断是：V2 fallback 数据没有显示出比 V1 更好的 val/test 潜力；由于 train IoU 也偏低，主要问题不应直接归因于“数据量还不够”。下一步应先排查 V2 target/mask、signal 语义、lift-off 定义、label area 和 runner/loss 适配，再决定是否扩大 COMSOL V2 数据规模。

## 54. S87-S90 COMSOL V2 target/signal diagnostics

S87 比较了 V1 S74 与 V2 S84 的 target / label / defect distribution。V1/V2 的 `mu_maps < 500` 与 `masks > 0.5` 完全一致，说明 target/mask 定义不是 V2 退化主因。但 V2 train mean label area ratio 为 `5.355850e-02`，V1 train 为 `1.172090e-01`，V2 只有 V1 的 `45.7%`；同时 V2 从单一 `ellipsoid` 变为 `rectangular_notch` / `rotated_rect` multi_defect，任务分布明显更难。

S88 比较了 Bz signal semantics。V2 train mean_abs_signal 是 V1 的 `3.689x`，mean_peak_abs_signal 是 V1 的 `11.747x`；V2 offset/peak 为 `0.041`，没有强 DC/background 主导。V2 lift-off peak abs 单调衰减比例 train / val / test 为 `0.92` / `0.95` / `0.90`，总体符合 lift-off 预期。V1 val/test signals 接近常量，说明 V1/V2 并不是完全同语义任务。

S89 因 S88 未发现强 offset 风险而跳过；S85 已使用 `per_sample_zscore`，额外 center-only probe 不是当前优先项。S90 的阶段判断是：V2 低于 V1 更可能来自 label area 更小、multi_defect / non-ellipsoid 任务更难，以及当前 runner/loss 对 small-label multi-component target 不适配；target/mask 定义不是主要瓶颈。
## 55. S91-S94 COMSOL V2 small-label runner adaptation

S91/S92 为 `train_conditional_dual.py` 增加了 small-label 诊断能力：`mask_bce_mode=bce|pos_weighted_bce|focal_bce`、`pos_weight`、`focal_gamma`、`focal_alpha`，以及 `point_sampling_mode=random|positive_balanced` 和 `positive_fraction`。默认配置保持旧行为。

S93 在 V2 COMSOL train=100 / val=20 / test=20 数据上运行三组 big 配置：

- `balanced_bce`
- `balanced_pos_weight5`
- `balanced_focal`

三组 train / val / test IoU 均为 `0.000000e+00`，`defect_area_pred` 也均为 `0`。这明显低于 S85 `big_multichannel_v2` 的 train / val / test IoU `3.023806e-01` / `2.593440e-01` / `2.768323e-01`。

Claude Code review 未发现 loss / sampling / label 对齐的 must-fix 实现错误。当前判断是：该结果更像 `positive_balanced` sparse point supervision 与 `mu_threshold` 输出路径的训练动态问题，而不是 schema 或 target 对齐问题。

S94 的阶段结论是：保留新增 runner 能力作为诊断工具，但不把 S93 配置作为 V2 默认训练策略。下一步应回到 S85 baseline，优先考虑 direct mask、area calibration、boundary-aware objective、curriculum 数据或模型/conditioning 调整。
## 56. S95-S98 COMSOL V2 train dynamics and curriculum bridge

S95 总结了 S91-S94 的失败模式：`balanced_bce`、`balanced_pos_weight5` 和 `balanced_focal` 均退化为全背景预测，train / val / test IoU 全为 `0`，`defect_area_pred` 全为 `0`。因此 simple imbalance loss 和当前 `positive_balanced` sampling 不是 V2 当前解法，S85 `big_multichannel_v2` 仍是较好的历史 baseline。

S96 为 `train_conditional_dual.py` 增加了 `training_history.csv` 和可选 pretrain / finetune curriculum。默认 `history_interval=0` 且无 pretrain 时保持旧行为。Claude Code review 未发现 runner / curriculum must-fix 问题。

S97 的 V2-only reproduce 和 V1 pretrain -> V2 finetune 都最终塌缩为全背景：

- `v2_only_baseline_reproduce` train / val / test IoU = `0.000000e+00` / `0.000000e+00` / `0.000000e+00`。
- `v1_pretrain_v2_finetune` train / val / test IoU = `0.000000e+00` / `0.000000e+00` / `0.000000e+00`。

training history 显示，V1 pretrain 本身可达到约 `0.53` batch IoU，但进入 V2 finetune 后仍被推向全背景。V2-only reproduce 也从非零预测面积逐步塌缩。因此当前问题不是单纯初始化或缺少 V1 warm start，而是 V2-specific train dynamics。

S98 的下一步策略是：不要继续盲目扫 focal / weighted BCE，也不要直接扩大 V2；优先处理 positive area / mask output dynamics，例如 area calibration、positive area prior、direct mask head、boundary-aware objective，或准备 V1-like -> intermediate -> V2-like 的 mixed curriculum 数据。

## 57. S99-S102 COMSOL V2 background-collapse suppression

S99 明确了 V2 当前的全背景塌缩问题：S93 的 small-label adaptation 和 S97 的 V1-to-V2 curriculum 都最终得到 `defect_area_pred=0`、IoU=0。V1 pretrain 本身可以拟合 V1，但进入 V2 finetune 后仍会塌缩，说明这更像 V2-specific train dynamics，而不是 target/mask 或简单 warm start 问题。

S100 在 `train_conditional_dual.py` 中新增 `area_loss_mode`、`lambda_area_loss` 和 `foreground_floor_ratio`，并在 `training_history.csv` 中记录 `area_loss`、`pred_area_soft_mean` 和 `true_area_mean`。默认 area loss 关闭，旧行为保持不变。

S101 测试了三组 V2 训练：`v2_baseline_with_history`、`area_ratio_mse` 和 `foreground_floor`。三组 train / val / test IoU 均为 `0`，hard `defect_area_pred` 也均为 `0`。`area_ratio_mse` 降低了连续 `mu` 误差，但没有让 hard foreground 越过 `mu_threshold`。

S102 的阶段结论是：简单 area calibration / foreground floor 不能单独解决 V2 全背景塌缩。当前瓶颈更接近 hard `mu_threshold` 输出路径、small-label / multi_defect 训练动态，以及缺少定位 / 边界约束。下一步应测试 direct mask + area loss、threshold margin 诊断、boundary/localization loss，或 staged curriculum。

## 58. S103-S106 COMSOL V2 threshold-margin diagnostics and repair

S103 诊断了 S101 的 hard-threshold collapse：三组 S101 run 都有 soft foreground 面积，但 hard `defect_area_pred=0`，且最后 history 的 `min_mu` 都仍高于 `500`。这说明 area loss 没有真正让 `mu_pred` 跨过 hard threshold。

S104 在 `train_conditional_dual.py` 中新增 threshold-margin loss：`threshold_margin_mode=positive_hinge|bidirectional_hinge`，并记录 positive / negative margin loss、sampled 正负点数量和 sampled 正负 `mu` 均值。默认关闭，旧行为保持不变。Claude Code review 未发现 must-fix 问题。

S105 的结果显示：positive-only margin 能恢复 hard foreground，但会退化为全前景；`bidirectional_margin_lambda1` 能恢复非零 IoU 并避免全前景，train / val / test IoU 为 `1.305192e-01` / `1.185913e-01` / `1.266814e-01`。

S106 的阶段判断是：hard threshold crossing 是真实瓶颈，但 threshold-margin 只解决 crossing，不解决定位 / shape。下一步应基于 bidirectional margin 测试 direct mask、boundary/localization loss 或 staged curriculum。

## 59. S107-S111 COMSOL V2 localization search early stop and quick gates

S107 总结了 S103-S106 的阶段结论：V2 已从“全背景塌缩”转向“hard foreground 可恢复但定位 / shape 仍差”的问题。hard threshold crossing 是必要条件，但 bidirectional margin 仍低于 S85 `big_multichannel_v2` baseline。

S108 在 `train_conditional_dual.py` 中加入 validation-aware best endpoint selection，支持 `val_selection_metric=none|eval_iou|eval_loss` 和 `val_selection_interval`。默认 `none` 保持旧行为；best state 只保存在内存中，不写出权重或 checkpoint。Claude Code review 已调用，interval guard 的 must-fix 已修复，smoke test 覆盖了 best-step history 记录。

S109 原计划 full V2 长实验包含 validation selection、margin+area 和 direct+area。当前已完成三组：`bidir_margin_val_select`、`bidir_margin_area_ratio` 和 `bidir_margin_floor`。其中 `bidir_margin_floor` 最好，train / val / test IoU 为 `1.508809e-01` / `1.344480e-01` / `1.440315e-01`，仍明显低于 S85。`direct_mask_area_ratio` 未运行，因阶段策略调整而停止。

S110 明确记录了 S109 partial / stopped 状态：继续跑剩余 full V2 配置预计信息增益低、耗时高。当前判断是 V2 不是简单 area / margin / endpoint selection 能解决，后续必须先通过 quick gates。

S111 建立 COMSOL V2 quick diagnostic gate protocol：Gate 1 为 5-sample train-overfit，Gate 2 为 20-train / 5-val mini generalization，Gate 3 才是 full V2 train / val / test。任何新 V2 objective / output path 必须先通过前一 gate，才允许进入下一规模。

## 60. S112-S116 COMSOL parametric inverse route

S112 将后续方向从 dense conditional mask loss 微调切换到 parametric inverse route。动机是 V2 dense mask runner 已在 weighted/focal/sampling/area/margin/validation selection 多轮诊断中表现为全背景、全前景或 localization 不足，而 V2 `defect_params` 提供了更低维、结构化的 geometry supervision。

新路线计划先从 multi-height Bz signals `[B,3,200]` / `[B,600]` 预测 component-level `presence`、`component_type`、`center_x`、`center_y`、`axis_x`、`axis_y`、`depth_or_shape_param` 和 `rotation_angle`，再用非可微 rasterization 评估 mask IoU / Dice。当前阶段只做 skeleton、smoke test 和 small train probe，不保存权重，不声称主线替代。

S113 已从 V2 train / val / test 的 `defect_params.csv` 构造 parametric targets。所有样本都有 3 个 component，无截断；`type_vocab` 在三个 split 中一致，为 `rectangular_notch, rotated_rect`；continuous schema 为 `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`。

S114 创建了 `ParametricInverseNet` skeleton，输入 `[B,600]`，输出 presence、type logits 和 continuous geometry parameters。S115 首个 V2 parametric inverse probe 显示该路线有非平凡信号：val / test rasterized param mask IoU 为 `0.369908` / `0.424462`，presence accuracy 均为 `1.0`，type accuracy 为 `0.65` / `0.6667`。

S116 的判断是：parametric route 比继续盲调 dense sparse-mask loss 更有希望，但当前仍是 pilot。主要瓶颈转为 rotation angle、component type 泛化、rasterization 近似和 signal encoder 表达能力。

## 61. S117-S120 COMSOL parametric raster oracle and refinement

S117 增加了 GT parametric rasterization oracle gate。`comsol_parametric_rasterizer.py` 使用 S113 的 ground-truth component targets rasterize mask，并与 S84 target masks 比较。train / val / test oracle avg IoU 为 `7.229967e-01` / `7.232882e-01` / `7.165838e-01`，三个 split 均超过 `0.70` gate，说明当前 parametric target + rasterizer 有足够上限继续训练诊断。

S118 将 parametric targets 从 raw `rotation_angle` 扩展为 `rotation_sin, rotation_cos`，并支持只用 train statistics 做 continuous target normalization。训练 runner 同步支持 normalized targets 的 raw-unit metric 反归一化、circular rotation MAE，以及 `type_class_weighting=inverse_freq`。

S119 运行 `refined_mlp` 后没有改善 held-out mask IoU。train / val / test param mask IoU 为 `6.689502e-01` / `3.257646e-01` / `3.885092e-01`；val / test rotation MAE 为 `7.278932e+00` / `7.859528e+00` degree；val / test type accuracy 为 `6.166667e-01` / `6.833333e-01`。由于 val/test mask IoU 低于 S115，`refined_mlp_longer` 被跳过。

S120 的判断是：parametric route 仍应继续，因为 oracle gate 通过且该路线避免了 dense V2 runner 的全背景 / 全前景塌缩；但当前瓶颈不再是简单 angle raw 表示或 target scale，而更可能是 signal encoder、component head、type/rotation loss 分解和 rasterization 近似。下一步应优先尝试 1D CNN / attention encoder、component-specific heads、grouped evaluation，并在进入 forward consistency / differentiable rasterization 前固定 rasterizer semantics。

下一阶段仍是 planned 状态，尚未完成：S121 做 parametric error decomposition，S122 测试 component-specific heads 与 stronger signal encoder，S123 做 quick gate architecture probe，S124 在 gate 通过后才做 best architecture full probe，S125 再给出 route decision。当前不建议回到 dense mask loss / margin / area / focal / sampling 的盲扫。

## 62. S121-S125 COMSOL parametric component heads and encoders

S121 拆解了 S115 raw 与 S119 refined 的 parametric errors。S115 train 已接近 oracle，train oracle gap 只有 `2.492512e-02`；但 val/test oracle gap 仍为 `3.533803e-01` / `2.921214e-01`。S119 refined 没有缩小 gap。当前主要 held-out 误差来自 type accuracy 约 `0.62` 到 `0.68`、rotation MAE 约 `7` 到 `8` degree，以及 geometry prediction 到 oracle upper bound 之间的泛化 gap。

S122 扩展了 parametric model 和 training runner：新增 `encoder_type=mlp|cnn1d|cnn1d_attention`、`head_mode=shared|component_specific`，以及 `lambda_center`、`lambda_axis`、`lambda_depth`、`lambda_rotation` continuous group loss weights。默认 `mlp/shared` 保持旧行为。

S123 quick gate 比较了四组 architecture。`raw_cnn_component_specific` 是 S123 内部最佳，val / test mask IoU 为 `3.662237e-01` / `3.892232e-01`，但仍未超过 S115 raw baseline 的 `3.699078e-01` / `4.244624e-01`。attention pooling 当前明显退化。

S124 对 `raw_cnn_component_specific` 做 6000-step full probe，val / test mask IoU 降到 `3.133465e-01` / `3.153288e-01`，因此不应作为默认配置。S125 的判断是 parametric route 继续成立，但当前最佳仍是 S115 raw MLP baseline；下一步应增加 per-sample prediction export、grouped diagnostics、slot decoder / set prediction 或 forward consistency，而不是继续拉长当前 CNN/component-specific 配置。

## 63. S126-S130 COMSOL parametric prediction diagnostics and set matching

S126 为 `train_comsol_parametric_inverse.py` 增加 `--export-predictions`，输出 train / val / test 的 per-sample / per-component prediction CSV，以及 per-sample mask IoU、Dice、oracle IoU、oracle gap 和 area metrics。S126 使用 S115 raw MLP 配置复现 baseline，train / val / test mask IoU 为 `6.980716e-01` / `3.699078e-01` / `4.244624e-01`，说明新增 export 不改变默认 fixed-order 行为。

S127 基于 S126 prediction exports 做 grouped diagnostics。presence 仍不是瓶颈；slot 1 的 type accuracy 相对较低，rotation error bin 与 mask IoU 明显相关，worst samples 存在较大的 oracle gap。当前主要问题仍是 held-out type / rotation / geometry generalization，而不是 target/mask schema 或 rasterizer upper bound。

S128 增加 `component_matching_mode=fixed|permutation_min`。`permutation_min` 对 `max_components=3` 枚举 6 个 permutation，选择最小 component loss 反传。Claude Code review 指出 prediction export 必须在 permutation 模式下记录 matching，且 padded slot 不应写有效误差；已修复并重跑 smoke / py_compile。

S129 比较 `fixed_reference` 与 `permutation_min`。fixed val / test mask IoU 为 `3.699078e-01` / `4.244624e-01`；permutation val / test mask IoU 下降到 `1.787238e-01` / `2.462286e-01`，type accuracy 和 rotation MAE 也劣化。S130 的判断是：loss-side permutation matching 不是当前 parametric route 的解法，当前最佳仍是 S115 raw MLP / shared head / fixed-order baseline。

下一步不建议回到 dense mask loss 小修，也不建议继续 loss-side permutation matching。更合理方向是 forward consistency / differentiable rasterization、geometry-aware rotation/type objective，或设计更明确的 slot/query decoder。

## 64. S131 COMSOL parametric set-matching stage summary

S131 收束了 S126-S130：prediction export 和 grouped diagnostics 已经能定位 sample/component 级错误；`permutation_min` 明显低于 fixed baseline，因此 loss-side set matching 不是当前主方向。Parametric route 继续，但下一步转向 differentiable raster mask supervision，让 mask-level error 直接反传到 geometry parameters，测试它是否能缩小 S115 与 S117 oracle upper bound 之间的 gap。

## 65. S132-S133 COMSOL differentiable raster supervision support

S132 新增 `comsol_differentiable_parametric_rasterizer.py`，实现 PyTorch soft rasterization。它沿用 S117 hard rasterizer 的 full width / full height axis 语义，使用 sigmoid soft rectangle 和 probabilistic union 生成 `[B,H,W]` soft mask。`rotation_angle` 按当前 raw target schema 视为 degree，`rotation_sin` / `rotation_cos` 使用 `atan2`。Claude Code review 指出原始 degree/radian heuristic 有风险，已修复为显式 schema 口径，并补充 smoke 覆盖。

S133 将 soft rasterizer 接入 `train_comsol_parametric_inverse.py`，新增 `lambda_raster_bce`、`lambda_raster_dice`、`raster_softness_cells` 和 `raster_target_source`。默认 raster loss 关闭，旧 parametric baseline 行为保持不变。当前 raster loss 仍是 mask supervision，不是 COMSOL forward consistency；是否改善 val/test mask IoU 由 S134 quick gate 判断。

## 66. S134-S135 COMSOL parametric raster-supervision probe

S134 比较 parameter-only baseline 与三组 differentiable raster supervision。`param_only_reference` val / test mask IoU 为 `3.699078e-01` / `4.244624e-01`；`raster_dice1` 为 `3.523885e-01` / `4.385081e-01`；`raster_bce05_dice1` 为 `3.697576e-01` / `4.096553e-01`；`raster_dice1_soft2` 为 `3.577784e-01` / `4.088950e-01`。

`raster_dice1` 小幅提升 test mask IoU，但 val 低于 baseline；其他 raster-supervised 配置也没有稳定超过 S115 / parameter-only baseline。中途发现 raster loss 需要对 normalized model output 反归一化后再 soft rasterize，已修复并重跑 raster-supervised runs，Claude Code review 确认无 must-fix。

S135 的判断是：differentiable raster supervision 有信号，但当前不应作为新默认配置。Parametric route 继续，当前最佳稳定配置仍是 S115 raw MLP / shared head / fixed-order baseline。下一步更适合测试两阶段训练、validation-aware raster fine-tune 或 forward consistency，而不是继续盲扫 raster loss 权重。

## 67. S136 COMSOL raster-supervision stage summary

S136 确认 raster loss 从头训练不作为默认：S134 的 raster-supervised runs 没有稳定超过 parameter-only baseline，`raster_dice1` 只小幅提升 test 但降低 val。因此下一步转向 two-stage parameter prefit + raster fine-tune，并配合 validation-aware endpoint selection，判断 raster loss 是否更适合作为后期 geometry / mask calibration。

## 68. S137-S139 COMSOL two-stage raster fine-tune

S137 在 `train_comsol_parametric_inverse.py` 中增加了 `--raster-loss-start-step`、`--val-selection-metric none|val_mask_iou|val_loss` 和 `--val-selection-interval`。默认行为保持旧 parameter-only 训练；best endpoint 只保存在内存中，不写出 checkpoint 或权重。Claude Code review 指出 delayed raster loss 与 `val_loss` selection 的 loss 组成不可比，以及 pre-raster endpoint 可能被保留；已修复为禁止该非法组合，并在 raster 阶段开始后重置 best tracking。

S138 比较三组 V2 raw parametric runs：`param_only_val_select`、`two_stage_raster_dice` 和 `two_stage_raster_bce_dice`。`param_only_val_select` 的 val mask IoU 提高到 `4.339882e-01`，但 test 降到 `3.966467e-01`；`two_stage_raster_dice` train mask IoU 提高到 `7.427683e-01`，但 val / test 只有 `4.050472e-01` / `4.022032e-01`；`two_stage_raster_bce_dice` 明显劣化到 val / test `2.810415e-01` / `3.105837e-01`。

S139 的判断是：two-stage raster fine-tune 能改善 train-side geometry / mask calibration，但没有稳定超过 S115 / S134 parameter-only baseline。当前最佳稳定配置仍是 S115 raw MLP / shared head / fixed-order baseline。Parametric route 继续，但下一步更适合 forward consistency / physics feature extraction，或 very short post-selection raster fine-tune；不应继续盲扫 raster BCE / Dice 权重。

## 69. S140 COMSOL raster fine-tune stage summary

S140 收束 S136-S139：raster loss 从头训练不稳定，two-stage raster fine-tune 没有稳定超过 S115，validation-aware selection 只改善 val endpoint而没有改善 test。当前最佳仍是 S115 / S134 raw MLP / shared head / fixed-order parameter-only baseline。

下一步不继续盲扫 raster BCE / Dice 权重，也不回到 dense conditional mask runner。当前更合理的方向是显式提取 multi-height MFL Bz 中的 peak、width、energy、lift-off decay ratio 和 inter-channel correlation 等 physics features，并与 parametric inverse route 做 feature fusion quick gate。

## 70. S141 COMSOL MFL physics features

S141 新增 `comsol_mfl_physics_features.py`，从 S84 COMSOL V2 train / val / test converted NPZ 中提取 peak、peak position、peak width、energy、abs area、center of abs mass、left/right balance、lift-off decay ratio 和 inter-channel correlation 等 physics-inspired features。

使用 `feature_mode=peak_decay_width` 后，train / val / test features shape 分别为 `[100,58]`、`[20,58]`、`[20,58]`，所有 features 均为 finite。主要 lift-off ratio 在 split 间稳定：`peak_abs_ch1_over_ch0` 均值约 `0.92-0.95`，`peak_abs_ch2_over_ch0` 均值约 `0.80-0.85`。下一步 S142 将这些 features 接入 parametric inverse runner，比较 `features_only` 和 `concat_latent`。

## 71. S142 COMSOL parametric feature fusion support

S142 扩展 `ParametricInverseNet`，新增 `feature_fusion_mode=none|features_only|concat_latent`。`features_only` 使用 `FeatureMLP` 编码 physics features；`concat_latent` 将 raw signal latent 与 feature latent concat 后再送入 heads。`train_comsol_parametric_inverse.py` 同步支持 `--feature-npz`、`--val-feature-npz`、`--test-feature-npz`，并只用 train features 计算 mean/std normalization。

默认 `feature_fusion_mode=none` 保持旧行为。Claude Code review 确认 feature sample alignment、train-only normalization、forward signature、默认兼容性和无 checkpoint 输出均无 must-fix。下一步 S143 比较 raw signal、features-only 和 raw+features。

## 72. S143 COMSOL physics feature fusion quick gate

S143 比较三组：`raw_signal_reference`、`physics_features_only` 和 `raw_plus_physics_features`。raw reference 复现 S115，train / val / test mask IoU 为 `6.980716e-01` / `3.699078e-01` / `4.244624e-01`。

`physics_features_only` 的 train mask IoU 达到 `7.273660e-01`，但 val / test 降到 `2.362846e-01` / `2.327774e-01`。`raw_plus_physics_features` 的 val type accuracy 和 rotation MAE 有改善，但 val / test mask IoU 只有 `3.313752e-01` / `3.051455e-01`。当前判断是：handcrafted physics features 有 train-side signal，但直接 features-only 或 concat fusion 没有稳定改善 held-out mask IoU；当前最佳仍是 S115 raw MLP baseline。

## 73. S144 COMSOL physics feature route summary

S144 的路线判断是：physics feature fusion 不应作为当前默认配置。`features_only` 说明 S141 features 能拟合 train，但泛化很弱；`concat_latent` 说明显式 features 对 val type / rotation 有局部帮助，但没有转化为 mask IoU 或 test 改善。

Parametric route 继续，当前最佳仍是 S115 / S143 raw MLP / shared head / fixed-order baseline。下一步优先考虑 forward consistency / learned forward surrogate，或把 physics features 作为 auxiliary prediction / regularization，而不是直接输入融合。

## 74. S145 COMSOL physics feature stage summary

S145 收束 S140-S144：raster fine-tune 未稳定改善，physics features 已成功提取，但 direct `features_only` / `concat_latent` fusion 没有超过 raw signal baseline。当前最佳仍是 S115 / S143 `raw_signal_reference` raw MLP / shared head / fixed-order parametric baseline，train / val / test mask IoU = `6.980716e-01` / `3.699078e-01` / `4.244624e-01`。

下一步不继续盲扫 raster loss、physics feature concat、CNN/attention encoder 或 dense mask runner。当前更合理的路线是训练 geometry -> multi-height Bz learned forward surrogate，并将其作为 forward consistency referee，用信号重建残差约束 inverse model 的 geometry prediction。

## 75. S146 COMSOL parametric forward surrogate implementation

S146 新增 `comsol_parametric_forward_surrogate.py` 和 `train_comsol_parametric_forward_surrogate.py`，用于学习 fixed-order component geometry -> multi-height Bz signal。`geometry_vector` 将每个 component 的 presence、type one-hot / probabilities 和 continuous geometry parameters 展平；surrogate 使用 MLP 输出 flattened Bz signal。

该 surrogate 训练时只使用 train split 的 signal z-score stats，并输出 normalized MSE、raw NRMSE、signal correlation、peak NRMSE 和 per-channel nrmse/corr。不保存 `.pt` / checkpoint。S146 只是 learned surrogate skeleton，是否足够作为 consistency referee 由 S147 gate 决定。

## 76. S147 COMSOL parametric forward surrogate quality gate

S147 使用 S84 V2 converted NPZ + S113 raw targets 训练 geometry -> Bz MLP surrogate。训练结果为 train / val / test `signal_nrmse_raw = 3.767854e-01 / 5.026852e-01 / 4.577952e-01`，`signal_corr = 9.258671e-01 / 8.657639e-01 / 8.886174e-01`。

S147 gate 通过：val/test correlation 均超过 `0.80`，且 raw NRMSE 均低于 `1.0`。因此继续 S148/S149，在 inverse training 中使用 in-memory frozen forward surrogate 做 consistency probe。该 surrogate 仍是 learned approximation，不等同 COMSOL solver。

## 77. S148 COMSOL parametric inverse forward-consistency support

S148 新增 `train_comsol_parametric_inverse_forward_consistency.py`。该 runner 先在内存中训练 geometry -> Bz surrogate，再冻结 surrogate 训练 inverse model；loss 为原 parametric loss 加 `lambda_forward_consistency * forward_signal_mse`。全过程不保存 surrogate 或 inverse 权重。

Claude Code review 指出 soft inverse probabilities 直接送入 hard-trained surrogate 会产生分布外输入；已修复为 straight-through hard presence 和 hard one-hot type，forward pass 与 surrogate 训练分布对齐，backward 仍通过 inverse probabilities / logits 传梯度。同时 continuous mean/std 已显式 detach，consistency target 直接复用 forward split 的 `signals_norm`。

## 78. S149 COMSOL parametric forward-consistency inverse probe

S149 完成三组比较。Parameter-only reference 复现 S115 / S143 baseline，train / val / test mask IoU = `6.980716e-01` / `3.699078e-01` / `4.244624e-01`。`lambda_forward_consistency=0.1` 的 train / val / test mask IoU 降为 `5.947000e-01` / `3.103259e-01` / `3.954980e-01`；`lambda=1.0` 进一步降为 `4.301722e-01` / `2.477730e-01` / `3.392353e-01`。

`lambda=0.1` 对 val/test rotation MAE 有局部改善，但没有改善 mask IoU；`lambda=1.0` 约束过强，伤害 type / geometry / mask。当前最佳仍是 S115 / S143 raw MLP / shared head / fixed-order parameter-only baseline。

## 79. S150 COMSOL forward consistency route summary

S150 的判断是：learned forward surrogate 通过了 signal-level quality gate，可以保留为 diagnostic referee，但当前 simple forward consistency objective 没有改善 parametric inverse mask IoU，不应设为默认。下一步如果继续 forward route，应先改 surrogate / residual 设计，例如 staged very-short consistency、dimension-specific residual 或 forward residual weighting；如果目标是直接提高 held-out mask IoU，更适合测试 physics-derived auxiliary prediction、type/rotation-specific supervision 或更强 geometry target。

## 80. S151 COMSOL forward consistency stage summary

S151 收束 S145-S150：forward surrogate 在 S147 中通过 signal-level gate，但 S149 的 simple forward consistency loss 没有超过 parameter-only baseline。`lambda_forward_consistency=0.1` 对 rotation MAE 有局部改善但降低 mask IoU，`lambda=1.0` 退化更明显。

当前不再盲扫 consistency lambda。Forward surrogate 暂时转为 residual diagnostic / consistency referee：先检查 residual 是否真的能区分 type / rotation / axis geometry 错误，再决定是否继续把它作为训练 loss。同时，下一步直接测试 type / rotation targeted supervision。

## 81. S152 COMSOL forward residual sensitivity diagnostic

S152 诊断显示 forward residual 对 rotation perturbation 非常敏感，对 type swap 中等敏感，但对 axis scaling 基本不敏感。val true / type-swapped / rotation-perturbed / axis-scaled / predicted avg signal NRMSE = `2.680198e-01` / `7.348850e-01` / `2.741137e+00` / `2.661547e-01` / `7.572894e-01`；test 对应为 `2.606982e-01` / `4.385888e-01` / `4.292691e+00` / `2.691335e-01` / `7.149153e-01`。

Predicted geometry residual 与 mask IoU 相关性有限，val / test 分别为 `-2.503890e-01` / `-4.378152e-01`。因此 learned forward residual 更适合作为 diagnostic referee，而不适合作为当前默认强训练 loss。下一步改为直接加强 type / rotation supervision。

## 82. S153 COMSOL type/rotation targeted supervision support

S153 在 `train_comsol_parametric_inverse.py` 中新增 targeted loss knobs：`lambda_type_extra`、`lambda_rotation_extra` 和 `rotation_loss_mode=mse|circular`。默认值均保持旧行为；当启用时，type CE 和 rotation-specific loss 会作为额外项加入 total loss，并记录到 metrics / history / run summary。

该改动用于直接验证 held-out type / rotation 是否能通过更强监督改善，而不是继续依赖 forward residual 这类间接 objective。

## 83. S154 COMSOL type/rotation targeted supervision probe

S154 四组 quick probe 显示：`param_only_reference` val / test mask IoU = `3.699078e-01` / `4.244624e-01`；`type_extra` = `2.987030e-01` / `4.268750e-01`；`rotation_extra` = `3.953165e-01` / `4.134323e-01`；`type_rotation_extra` = `3.817137e-01` / `3.627724e-01`。

`type_extra` 没有改善 type accuracy，test mask IoU 的微小提升不足以作为稳定改善；`rotation_extra` 改善 train rotation fit 和 val mask IoU，但 test 低于 S115；组合 loss test 退化更明显。当前仍没有稳定超过 S115 / S143 parameter-only baseline 的配置。

## 84. S155 COMSOL forward residual and type/rotation decision

S155 的决策是：forward residual 保留为 diagnostic / ranking 工具，不作为当前默认 loss；simple type / rotation extra losses 也不作为默认。当前最佳稳定配置仍是 raw MLP / shared head / fixed-order parameter-only baseline。

下一步更适合检查 type / rotation 数据平衡、生成更均衡 COMSOL data、引入 type-specific heads / auxiliary classifier，或转向更强 target representation，例如 Piao-style profile parameterization，而不是继续调同类 loss 权重。

## 85. S156-S157 COMSOL parametric oracle ablation setup

S156 收束 S151-S155：forward residual 和 simple type / rotation extra losses 都没有形成稳定新 best，当前最佳仍是 S115 / S143 / S154 raw MLP / shared head / fixed-order parameter-only baseline。继续调同类 loss 权重的信息密度已经偏低。

S157 因此新增 parameter-level oracle ablation diagnostic：从 S126 per-component predictions 出发，逐项用 S113 GT 参数替换 type、rotation、center、axis、depth 和 continuous，再复用 S117 hard rasterizer 评估 mask IoU / Dice。该脚本不训练、不保存权重、不生成图片；rotation 按 raw degree 处理，并转成 sin/cos schema 后 rasterize，避免旧 angle heuristic。

## 86. S158-S159 COMSOL parametric oracle ablation results

S158 的 `pred_all` 复现 S115 baseline，train / val / test mask IoU 为 `6.980716e-01` / `3.699078e-01` / `4.244624e-01`。`gt_all` 精确对齐 S117 oracle，train / val / test 为 `7.229967e-01` / `7.232882e-01` / `7.165838e-01`，说明 prediction CSV、S113 targets 和 hard rasterizer 对齐可信。

Oracle ablation 显示最大瓶颈不是 type 或 rotation，而是 center localization。`gt_center` 将 val / test mask IoU 提升到 `7.148715e-01` / `7.229199e-01`，几乎关闭了 baseline 到 oracle 的主要 gap。`gt_axis` 只有小幅提升，`gt_rotation` 对 val/test 没有改善，`gt_type` 与 `gt_depth` 在当前 hard rasterizer 下不直接改变 mask。

S159 的决策是：parametric route 继续，但下一步应围绕 center / localization targeted diagnostic，例如 center-specific loss scaling、coordinate reparameterization、center-bin classification + offset 或 Bz peak-position alignment features；不继续盲扫 type / rotation / forward consistency loss。

## 87. S160-S161 COMSOL center bottleneck diagnostics

S160 将 S158/S159 的 oracle ablation 结论收束为 center-localization stage：`gt_center` 将 val / test mask IoU 从 `0.369908` / `0.424462` 提升到 `0.714872` / `0.722920`，接近 S117 oracle，因此 type / rotation / forward / raster loss 不再是当前优先项。

S161 新增 `comsol_parametric_center_diagnostics.py`，从 S126 predictions、S113 targets 和 S84 x/y grid 计算 center error。Val / test `center_l2_grid_mae` 为 `8.017750` / `6.998191`，`center_axis_relative_l2_mae` 为 `0.445062` / `0.394215`；center error 与 mask IoU 在 val/test 上强负相关，Pearson 约 `-0.927` / `-0.911`，Spearman 约 `-0.902` / `-0.908`。

## 88. S162 COMSOL center-aware loss support

S162 在 `train_comsol_parametric_inverse.py` 中新增 `--lambda-center-grid`、`--lambda-center-axis-relative` 和 `--center-axis-relative-eps`。Grid loss 将反归一化 center error 按 x/y grid spacing 换算为 cell error；axis-relative loss 用 GT full-width/full-height axis 做尺度归一。两者都只对 present components 生效，默认值为 0，保持旧行为。

Claude Code review 指出 grid spacing tolerance 过严和 eval center metric 缺少 guard 两个 must-fix；已修复并重跑 smoke / py_compile。

## 89. S163-S164 COMSOL center-aware loss probe

S163 采用同一轮 1500-step reference。`param_only_1500_reference` val / test mask IoU = `0.415913` / `0.427099`；`center_grid_loss` 提升到 `0.470171` / `0.504389`，同时降低 val/test center grid 和 axis-relative error；`center_axis_relative` 退化到 `0.374787` / `0.392738`。因此只有 `center_grid_loss` 通过 gate。

S164 对 `center_grid_loss` 做 3000-step confirm，train / val / test mask IoU = `0.726483` / `0.469423` / `0.498874`，超过 S115 / S158 pred_all baseline 的 `0.698072` / `0.369908` / `0.424462`。这说明 center-aware grid loss 是当前第一个稳定改善 val/test parametric mask IoU 的 targeted change。

## 90. S165 COMSOL center-localization decision

当前最佳 parametric 配置更新为 raw MLP / shared head / fixed-order + `lambda_center_grid=0.1`。该配置仍低于 oracle，但已验证 center localization 是可行动瓶颈。下一阶段不应继续简单加大 lambda，而应测试 center-bin classification + offset、signal-to-center auxiliary head、per-component peak-position alignment features 或 stable repeat。
## 91. S166-S167 COMSOL center-grid stability setup

S166 将 S160-S165 收束为 stability validation：`lambda_center_grid=0.1` 已经在 S163/S164 形成 val/test 改善，但仍需多 seed repeat 才能作为当前 parametric route candidate。S167 因此给 `train_comsol_parametric_inverse.py` 增加 `--seed`，默认 `0`，并让 `torch`、`numpy` 和 Python `random` 都使用该 seed；metrics 和 run summary 均记录 seed。

Claude Code review 指出 Python `random` 没有 seed 是 must-fix；已修复并重跑 smoke / py_compile。默认 `--seed=0` 保持旧命令兼容。

## 92. S168-S170 COMSOL center-grid stability result

S168 复用 S164 作为 `existing_unrecorded`，新增 seed1 和 seed2。三次 center-grid runs 的 val / test mask IoU 分别为：`0.469423` / `0.498874`、`0.485716` / `0.505590`、`0.446966` / `0.503713`。Seed1 没有触发 early stop，因此 seed2 正常执行。

S169 所有 acceptance criteria 均为 true：每个 run 的 val/test IoU 都高于 historical param-only baseline `0.369908` / `0.424462`，median test IoU 为 `0.503713`，3/3 runs test IoU >= `0.48`，且 val/test center_grid_mae 均低于 S161 baseline。

S170 决策：raw MLP / shared head / fixed-order + `lambda_center_grid=0.1` 升级为当前 COMSOL parametric route candidate。下一步不继续 lambda sweep，而是以该 candidate 为 reference，若继续提升则转向 center-bin classification + offset、signal-to-center auxiliary head 或 peak-position alignment。

## 93. S171-S175 COMSOL center-grid candidate consolidation

S171-S175 is a documentation-only consolidation stage. It promotes raw MLP / shared head / fixed-order + `lambda_center_grid=0.1` as the current COMSOL parametric route candidate on `feature/dual-network-variational`, while keeping the branch boundary explicit: this is not a main baseline replacement.

`DUAL_NETWORK_REPRODUCE.md` now includes an explicit candidate command with `--lambda-center-grid 0.1`, `--lambda-center-axis-relative 0.0`, `--seed <N>`, raster loss disabled, forward consistency absent, and validation selection disabled. The CLI defaults remain unchanged, no Python was modified, and no new training or seed run was added.

The next recommended route is center-bin classification + offset. The stage intentionally does not continue center lambda sweeps, `center_axis_relative`, type/rotation loss, forward consistency, or raster loss.

## 94. S176-S180 COMSOL center-bin offset route

S176-S177 added optional center-bin + offset support while preserving the old default `continuous` center representation. In `bin_offset` mode the model predicts per-axis center bins plus bin-normalized offsets, decodes them back to `center_x` / `center_y`, and uses the decoded center for center-grid loss, evaluation, prediction export, and rasterization.

S178 quick gate compared the same-seed current candidate reference against `center_bin_offset` and `center_bin_offset_plus_grid`. The winner was `center_bin_offset_plus_grid`, with val/test mask IoU `0.546311` / `0.586546` versus the same-round reference `0.494508` / `0.493461`; val/test center_grid_mae also decreased.

S179 3000-step confirm for `center_bin_offset_plus_grid` reached train / val / test mask IoU `0.716101` / `0.542935` / `0.581320`, with val/test center_grid_mae `3.362513` / `2.721649`. S180 therefore continues the center-bin route, but does not yet replace the S170 candidate until multi-seed stability is verified.

## 95. S181-S185 COMSOL center-bin offset plus grid stability

S181-S185 reused S179 seed1 and added seed2/seed3 for the fixed `center_bin_offset_plus_grid` configuration: raw MLP / shared head / fixed-order, `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, and `lambda_center_grid=0.1`. No Python code or CLI defaults were changed.

The three val / test mask IoU pairs were `0.542935` / `0.581320`, `0.484303` / `0.575504`, and `0.492127` / `0.578738`. All runs exceeded the S170 center-grid candidate test range and had val/test `center_grid_mae` below the S170 worst values. S184 acceptance criteria all passed, so S185 promotes `center_bin_offset_plus_grid` to the current COMSOL parametric route candidate for this branch.

Boundary: this is not a main baseline replacement. Seed2/seed3 pass the gate but have lower val IoU than S179 seed1, so the next center-bin stage should continue observing validation stability rather than treating the configuration as a final solution.

## 96. S186-S188 COMSOL center-bin candidate consolidation and diagnostics

S186 formally consolidated `center_bin_offset_plus_grid` as the current COMSOL parametric route candidate on `feature/dual-network-variational`: raw MLP / shared head / fixed-order, `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, and `lambda_center_grid=0.1`. This remains branch-local, is not a main baseline replacement, and is not a final solution.

S187 used only existing S179/S183 prediction exports and metrics to diagnose remaining errors. Test behavior is stable: test IoU stays in `0.575504-0.581320` and test `center_grid_mae` stays in `2.721649-2.929023`. Val is less stable because seed2/seed3 val `center_grid_mae` rises to `6.282760` / `6.026593` versus seed1 `3.362513`; the likely remaining bottleneck is x-bin stability first, with y-bin secondary.

S188 selects `signal-to-center auxiliary head` as the next unique route. The stage does not recommend more lambda tuning, `center_axis_relative`, COMSOL V3 generation, type/rotation loss sweeps, raster/forward-consistency sweeps, or dense conditional mask runner work.

## 97. S189-S193 COMSOL signal-to-center auxiliary head

S189-S193 tested an optional signal-to-center auxiliary head on top of the current S185 `center_bin_offset_plus_grid` branch candidate. S190 adds the aux head with default-off CLI knobs and preserves the existing center-bin candidate behavior when aux is disabled.

S191 used a same-round 1500-step seed-1 reference. The reference reproduced a strong candidate-level result with val/test IoU `0.546311` / `0.586546`. `aux_center_bin_offset` reached `0.516648` / `0.567790`, and `aux_center_bin_offset_xweighted` reached `0.542723` / `0.580217`. Both auxiliary variants were below the same-round reference and worsened held-out `center_grid_mae`.

S192 was therefore skipped. S193 keeps the current COMSOL parametric route candidate unchanged: raw MLP / shared head / fixed-order with `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, and `lambda_center_grid=0.1`. The auxiliary head is not promoted and should not be followed by another simple lambda sweep.

## 98. S194-S197 COMSOL center-bin sample-level failure diagnostics

S194-S197 did not run new training and did not modify the training runner. S195 added a read-only diagnostic script that reconstructs per-component x/y bin correctness, offset error, center-grid error, grouped diagnostics, and worst samples from existing prediction exports.

S196 applied that diagnostic to S191 runs. The current candidate reference has val/test x wrong rates `0.200000` / `0.133333`, higher than y wrong rates `0.083333` / `0.033333`. Slots 0 and 2 are more fragile than slot 1, and low-IoU samples have much larger center-grid error. The auxiliary variants did not help: plain aux worsened both x/y errors, and x-weighted aux only slightly reduced val x wrong rate while worsening y wrong rate, held-out center-grid error, and IoU.

S197 therefore keeps S185 `center_bin_offset_plus_grid` as the current branch COMSOL parametric candidate and recommends the next route as center-x-bin focused calibration / hard-sample refinement of the main center-bin output, not another auxiliary-head or lambda sweep.

## 99. S198-S202 COMSOL x-bin calibration quick gate

S198-S202 tested a narrow version of the S197 recommendation: optional reweighting of the main center-bin CE loss, without changing model structure, enabling the auxiliary head, or using dynamic hard-sample mining. S199 adds `--center-bin-x-weight`, `--center-bin-y-weight`, and `--center-bin-slot-weights`; defaults keep the old main center-bin loss numerically equivalent to `0.5 * (x_loss + y_loss)`.

S200 used a same-round 1500-step seed-1 reference with explicit prediction exports, then ran `comsol_center_bin_failure_diagnostics.py` on all three outputs. The reference reproduced a strong candidate-level result with val/test IoU `0.546311` / `0.586546`. `x_bin_weighted` reached `0.545284` / `0.555791`; `x_bin_slot_weighted` reached `0.518272` / `0.543191`. The x-only run slightly reduced val x wrong rate, but worsened test x wrong rate, held-out center_grid_mae, and test IoU. Slot-aware weighting degraded both held-out splits.

S201 was skipped because no S200 group passed gate. S202 keeps S185 `center_bin_offset_plus_grid` as the current branch COMSOL parametric candidate and stops simple x-bin / slot-weight sweeps. The next useful route is diagnostic: inspect low-IoU samples where bins are correct but offset, decoded center, or geometry interaction still fails.

## 100. S203-S206 COMSOL hard-sample / data-design package

S203-S206 freezes the S185 `center_bin_offset_plus_grid` candidate and turns the S198-S202 negative result into a diagnostic/data-design stage. No training was run, no seed was added, and no Python training or model logic was changed.

S204 packages existing S200 exports into hard-sample tables. The current-candidate hard set has 23 held-out sample keys, with labels `x_bin_wrong=12`, `both_bins_wrong=3`, `bins_correct_center_or_offset_bad=5`, `geometry_or_type_interaction=2`, and `y_bin_wrong=1`. This mixed taxonomy means the residual issue is not solved by simply increasing x-bin CE or slot weights. Some samples need better x-bin / slot-boundary data coverage, while bins-correct low-IoU cases need decoded-center, offset, or geometry-interaction diagnosis.

S205 drafts a V2-compatible hard-case COMSOL data request focused on x-bin boundaries, slot 0 / slot 2 fragility, near/medium component spacing, mixed type sequences, rotated rectangles, and all-bins-correct low-IoU cases. S206 recommends using that data-design artifact plus bins-correct low-IoU diagnostics before any future training; the S185 branch candidate remains unchanged and is not a main baseline replacement.

## 101. S207 COMSOL V3 hard-case pack preflight

S207 turns the hard-case request into a concrete pilot spec. The default request is `60/20/20` train/val/test, with fallback `30/10/10` if COMSOL generation cost is high. The default mix follows S204's mixed taxonomy: `x_bin_wrong_like` `24/8/8`, `both_bins_wrong_like` `9/3/3`, `bins_correct_center_or_offset_bad` `15/5/5`, `geometry_or_type_interaction` `9/3/3`, and `rare_y_bin_wrong` `3/1/1`.

The pack remains V2-compatible: `signals_multiheight.csv`, `targets.npz`, `defect_params.csv`, and split `README.md`; three Bz channels, signal length `200`, `lift_off_values=[0.5,1.0,2.0]`, and component metadata compatible with `comsol_parametric_targets.py`. S207 does not generate data or run training. After COMSOL generation, the next branch step should be an ingest gate before any model run.

## 102. S208-S211 COMSOL V3 hard-case ingest gate

S208-S211 ingested the real COMSOL V3 hard-case fallback pack into the branch experiment tree without running training. The pack was copied into `experiments/dual_network/S208_comsol_v3_hard_case_ingest/raw/`, converted to V2-compatible NPZ files, and checked for shape, finite values, complete `x_index`, hard-case label coverage, and `masks == (mu_maps < 500)`.

Converted train/val/test signal shapes are `[30,3,200]`, `[10,3,200]`, and `[10,3,200]`. The hard-case distribution is train `10/5/7/5/3`, val `3/2/2/2/1`, and test `3/2/2/2/1` for `x_bin_wrong_like`, `both_bins_wrong_like`, `bins_correct_center_or_offset_bad`, `geometry_or_type_interaction`, and `rare_y_bin_wrong`.

S209 built fixed-order parametric targets with schema `center_x`, `center_y`, `axis_x`, `axis_y`, `depth_or_shape_param`, and `rotation_angle`; `type_vocab=rectangular_notch`. S210 oracle rasterization reached train/val/test IoU `1.000000` / `1.000000` / `1.000000`, so the ingest gate passes. S211 recommends evaluating the frozen S185 `center_bin_offset_plus_grid` candidate on V3 hard-case val/test next. Boundary: this fallback pack is real COMSOL data, but currently only single rectangular Block solves, not true rotated or multi-component COMSOL geometry.

## 103. S212-S216 COMSOL V3 hard-case candidate evaluation

S212-S216 evaluated the current S185/S181 `center_bin_offset_plus_grid` branch candidate on the real COMSOL V3 hard-case fallback pilot. The intended V2-trained zero-shot evaluation did not produce valid metrics because the V3 center targets are outside the V2 train center-bin grid: V2 uses meter-scale `[-0.04,0.04]` / `[-0.01,0.01]`, while V3 uses `[0,4500]` / `[0,3000]`.

S214 therefore tested only same-grid V3 train -> V3 val/test learning. The center-bin candidate reached train/val/test IoU `0.019715` / `0.046905` / `0.044968`; the continuous param-only reference reached `0.038119` / `0.078177` / `0.036448`. Both are too weak to treat as useful V3 hard-case learning.

S215 grouped the available prediction exports by `hard_case_type`. Failure is broad rather than concentrated in one class: `bins_correct_center_or_offset_bad` is consistently among the worst held-out groups, but other hard-case types also have near-zero IoU. S216 decision: keep S185 as the current branch candidate for V2-style data, but do not claim it is validated on this V3 fallback pack. The next step is to fix V3 geometry coordinate conventions or add an explicit V3-to-V2 geometry-unit conversion before rerunning ingest/oracle and candidate evaluation.

## 104. S217-S221 COMSOL V3 geometry convention normalization

S217-S221 fixes the coordinate convention issue revealed by S213. S217 confirms that V3 raw data is internally self-consistent but not V2-compatible: V2 uses centered meter-scale coordinates `[-0.04,0.04] / [-0.01,0.01]`, while V3 raw uses COMSOL model coordinates `[0,4500] / [0,3000]`.

S218 creates a normalized V3 copy without modifying S208 raw. It maps x/y and defect center/axis x/y into the V2 meter-scale convention while leaving `signals`, `mu_maps`, and `masks` unchanged. Depth / z is deliberately not scaled because current mask rasterization does not use depth and the V3 z convention remains unaudited.

S219 rebuilds normalized parametric targets and reaches train/val/test oracle IoU `1.000000` / `1.000000` / `1.000000`. S220 then verifies runability: V2 train to normalized V3 val/test no longer triggers the center-bin range error. S221 keeps S185 as the current branch candidate and defers actual V3 performance evaluation to a later normalized-V3 evaluation stage.

## 105. S222-S226 COMSOL normalized V3 hard-case candidate evaluation

S222-S226 evaluates the S185 `center_bin_offset_plus_grid` branch candidate on the normalized V3 hard-case fallback pilot. S223 V2-trained zero-shot now runs, but val/test mask IoU is only `0.002348` / `0.012360`, with val/test `center_grid_mae` `58.098274` / `61.760284` and essentially no x-bin accuracy.

S224 normalized V3 train quick probe is also weak. The candidate reaches train/val/test IoU `0.019538` / `0.047127` / `0.044771`; the continuous param-only reference reaches `0.039498` / `0.080140` / `0.037464`. The current candidate is therefore not validated on this normalized V3 fallback pilot.

S225 grouped diagnostics show `bins_correct_center_or_offset_bad` as the most consistent hardest group, but failure is broad across hard-case types. S226 keeps S185 as the current V2-style branch candidate and recommends a larger real COMSOL V3 hard-case pack with true rotated and multi-component geometry coverage before further model/loss changes.

## 106. S227-S231 COMSOL normalized V3 tiny-overfit precondition gate

S227-S231 reframes the normalized V3 failure as a train-learnability and signal-target sanity problem. Target-side checks pass: masks match `mu_maps < 500`, bbox centers align with normalized defect centers, S219 targets match normalized defect parameters, and center-bin targets are in range.

The blocker is signal scale. S227 shows every train/val/test normalized V3 sample triggers the runner `std < 1e-8` signal floor; train signal std is only `4.734403e-10` to `9.312454e-09`. Therefore S228 one-sample, S229 five-sample, and S230 full-train fit gates were skipped before training.

S231 decision: do not generate a larger V3 pack and do not change the model/runner yet. The next step is to inspect COMSOL V3 Bz signal export semantics, probe height, field expression, source/magnetization scaling, lift-off extraction, and the runner normalization floor.

## 107. S232 COMSOL V3 repaired Bz signal 3-sample smoke

S232 validates the repaired V3 Bz signal export route on three real COMSOL smoke samples without training and without generating a fallback pack. The repaired route uses a near-defect probe and anomaly / delta-Bz signal instead of raw absolute Bz.

The three hard-case labels are `x_bin_wrong_like`, `bins_correct_center_or_offset_bad`, and `rare_y_bin_wrong`. The smoke output has `1800` signal rows, converted shape `[3,3,200]`, finite values, complete per-channel `x_index=0..199`, and `masks == (mu_maps < 500)` mismatch `0`.

Per-sample/channel std stays above `1e-8` with range roughly `9.85e-07` to `1.05e-06`, and peak-to-peak stays above `1e-8` with range roughly `4.51e-06` to `7.83e-06`. This validates the repaired signal export route across x-bin, offset/axis, and y-offset smoke cases. Next step: generate a repaired V3 hard-case fallback pack; this smoke is not a training or model-evaluation result.

## 108. S233-S236 COMSOL repaired V3 hard-case ingest gate

S233-S236 ingests the repaired V3 hard-case fallback pack generated with per-sample fresh COMSOL models and repaired near-defect reduced-field Bz signals. The distribution gate uses the ingested `defect_params.csv`, not external summary text: train is `10/5/7/5/3`, val is `3/2/2/2/1`, and test is `3/2/2/2/1` for `x_bin_wrong_like`, `both_bins_wrong_like`, `bins_correct_center_or_offset_bad`, `geometry_or_type_interaction`, and `rare_y_bin_wrong`.

Converted shapes are `[30,3,200]`, `[10,3,200]`, and `[10,3,200]`. Signal std ranges are train `1.678613e-06`-`3.162381e-06`, val `1.943240e-06`-`2.826707e-06`, and test `2.101787e-06`-`2.876971e-06`; peak-to-peak ranges are train `7.409328e-06`-`1.984153e-05`, val `1.030256e-05`-`1.845700e-05`, and test `8.955499e-06`-`1.965699e-05`. All values are finite, `x_index` is complete, and `masks == (mu_maps < 500)` mismatch is `0`.

S234 target schema is `center_x`, `center_y`, `axis_x`, `axis_y`, `depth_or_shape_param`, and `rotation_angle`, with `type_vocab=rectangular_notch`. S235 oracle rasterization reaches train/val/test IoU `1.000000` / `1.000000` / `1.000000`, so the repaired V3 ingest gate passes. This stage does not train or evaluate candidate strength; the next stage should do branch-local repaired V3 candidate evaluation.

## 109. S237-S241 COMSOL repaired V3 hard-case candidate evaluation

S237-S241 evaluates the current S185 `center_bin_offset_plus_grid` candidate on the repaired V3 fallback pilot. S238 V2-train to repaired-V3 zero-shot does not produce metrics because the repaired V3 pack remains in raw COMSOL coordinates and still trips the train-grid center-bin check: `center_x target is outside the x grid range`. This is separate from the old near-constant signal failure.

S239 same-grid repaired V3 training shows the repaired signal is learnable on train. The candidate reaches train/val/test IoU `0.998851` / `0.052874` / `0.197143`; param-only reaches `0.986927` / `0.000000` / `0.157851`. The candidate's held-out center-grid error remains high at val/test `14.233663` / `13.523300`, with y-bin accuracy only `0.100000` / `0.300000`.

S240 grouped diagnostics show near-perfect train groups but broad held-out failure. Val has nonzero mean IoU only for `both_bins_wrong_like`; test is hardest for `geometry_or_type_interaction` and `rare_y_bin_wrong`. S241 therefore does not validate the current candidate on repaired V3 held-out data. The next unique recommendation is a larger repaired V3 hard-case pack before mixed V2+V3 training or multi-seed candidate validation.

## 110. S242-S246 COMSOL repaired V3 coordinate normalization and reevaluation

S242-S246 fixes the repaired V3 coordinate convention before interpreting candidate performance. S242 maps repaired V3 raw COMSOL coordinates `[0,4500]` / `[0,3000]` to the V2-compatible `[-0.04,0.04]` / `[-0.01,0.01]` grid and applies the same affine transform to defect center x/y and axis x/y. Signals, masks, and `mu_maps` are unchanged; depth/z stays raw because it is not used by the current 2D mask rasterizer and the z convention is still unaudited.

S243 confirms the normalized pack remains aligned: train/val/test oracle IoU is `1.000000` / `1.000000` / `1.000000`. S244 confirms V2-train to normalized repaired V3 val/test now runs, but zero-shot IoU is only `0.007616` / `0.005248`.

S245 shows the normalized repaired V3 train split is learnable, with current candidate train IoU `1.000000`; held-out val/test remain weak at `0.055172` / `0.188341`. The continuous param-only reference reaches `0.773635` / `0.000000` / `0.171178`. S246 keeps the current S185 branch candidate unchanged and recommends a larger repaired V3 hard-case pack before mixed V2+V3 training or candidate promotion.

## 111. S247-S253 COMSOL V3 polygon geometry route

S247-S253 starts the long-term polygon / corner-point route for true V3 geometry. The key mechanism is that V3 raw coordinates `[0,4500] / [0,3000]` are mapped to the V2-compatible grid by different x/y scales, so a raw-space rotated rectangle is not generally representable by the old normalized `center + axis + rotation` schema. This is an oracle representation ceiling, not a model-training issue.

S248 defines the COMSOL polygon export contract with `polygon_params.csv` containing actual COMSOL geometry vertices in both raw and normalized coordinates. S249 adds a fixed four-corner polygon target builder and a hard point-in-polygon rasterizer, without touching the inverse training runner.

S250 mock oracle smoke passes, and S251-S252 true COMSOL 3-sample smoke passes with per-sample polygon oracle IoU `1.000000`. The smoke covers true rotated geometry and a true two-component Union. S253 therefore recommends continuing the polygon route before any V3 true-geometry model training. The S185 `center_bin_offset_plus_grid` candidate remains the current V2-style branch candidate and is not a main baseline replacement.

## 112. S254-S258 COMSOL V3 polygon hard-case ingest gate

S254-S258 ingests the polygon-compatible repaired V3 hard-case pack generated from real COMSOL solves. Converted train/val/test shapes are `[30,3,200]`, `[10,3,200]`, and `[10,3,200]`; signal values are finite and non-near-constant, hard-case distributions match `10/5/7/5/3`, `3/2/2/2/1`, and `3/2/2/2/1`, and `masks == (mu_maps < 500)` mismatch is `0`.

Polygon targets use `polygon_vertices_raw`, `polygon_vertices_norm`, `polygon_vertex_mask`, and `polygon_presence` with `max_components=3` and `max_vertices=4`. The polygon oracle gate reaches train/val/test mean and min IoU `1.000000`, so the pack is ready for polygon inverse model planning. No training was run and the S185/S181 center-bin candidate remains unchanged.

## 113. S259-S263 COMSOL V3 polygon inverse first gate

S259-S263 adds an independent polygon inverse model and runner instead of extending the old parametric runner. The model predicts `presence_logits [B,3]`, `type_logits [B,3,T]`, and `vertices_norm [B,3,4,2]`; the runner trains with presence BCE, present-slot type CE, and present-vertex SmoothL1, while hard polygon rasterization is evaluation-only.

Model and runner smoke tests pass. The one-sample overfit gate on train sample `0` reaches presence/type accuracy `1.000000` and normalized vertex MAE `4.207401e-05`, but hard polygon mask IoU is only `0.883178`, below the stop threshold `0.90`. Therefore 5-sample and train30 quick probes are skipped. The S185/S181 center-bin candidate remains unchanged; next step is vertex-to-raster sensitivity diagnostics.

## 114. S264-S268 COMSOL V3 polygon one-sample raster sensitivity repair

S264-S268 diagnoses and repairs the S262 one-sample polygon inverse stop condition without entering 5-sample or train30. S265 confirms the target and rasterizer are aligned: target-vertex oracle IoU remains `1.000000`, while the S262 prediction adds `25` false-positive pixels. The largest predicted vertex errors are only `0.333211` x-cells and `0.494064` y-cells, but that is enough to expand the hard-rasterized mask from `189` to `214` pixels.

S266 adds default-off repair support to the polygon runner: `--vertex-loss-space norm|grid`, `--lambda-area-aux`, and `--lambda-edge-aux`. Defaults preserve the S259-S263 behavior, and no differentiable polygon rasterizer or model-structure change is introduced.

S267's first repair run, `longer_overfit`, passes the one-sample gate with train IoU `1.000000`, presence/type accuracy `1.000000`, vertex MAE `7.786439e-07`, and pred/target area `189` / `189`. Grid-loss and area/edge variants are therefore skipped. S268 keeps the S185/S181 center-bin candidate unchanged and recommends resuming the staged 5-sample polygon inverse gate next.

## 115. S269-S273 COMSOL V3 polygon 5-sample overfit gate

S269-S273 runs only the polygon inverse 5-sample overfit gate. The subset uses source train samples `0,11,15,22,27`, covering all five hard-case labels: `x_bin_wrong_like`, `both_bins_wrong_like`, `bins_correct_center_or_offset_bad`, `geometry_or_type_interaction`, and `rare_y_bin_wrong`.

The 5-sample overfit passes with mean/min train polygon IoU `0.996028` / `0.985401`, presence accuracy `1.000000`, present type accuracy `1.000000`, and vertex MAE `5.359486e-06`. The worst sample is source sample `27` (`rare_y_bin_wrong`) with IoU `0.985401` and area diff `-2` pixels. Both multi-component samples reach IoU `1.000000`.

S273 clears the 5-sample gate but does not promote a polygon inverse candidate and does not replace the S185/S181 center-bin branch candidate. The next staged step should be a train30 / val10 / test10 polygon quick probe.

## 116. S274-S278 COMSOL V3 polygon train30 quick probe

S274-S278 runs the first train30 / val10 / test10 polygon inverse quick probe using the S254-S258 polygon V3 pack and the S271 successful overfit configuration. This stage does not change model structure or runner code, does not generate new COMSOL data, and does not run multi-seed validation.

The train30 gate fails. Train mean/min polygon IoU is `0.731445` / `0.518519`, below the `0.90` / `0.80` acceptance gate. Train presence/type accuracy is `1.000000` / `1.000000`, and train vertex MAE is `1.793932e-04`, so the model learns component existence and type but not enough vertex precision across all 30 train samples.

Val/test are observation-only and are weak: mean IoU `0.033122` / `0.089484`. S276 shows broad train-fit weakness across hard-case types, with `x_bin_wrong_like` best at mean IoU `0.803422` and `geometry_or_type_interaction` / `bins_correct_center_or_offset_bad` weakest at `0.654147` / `0.682215`. The next stage should plan targeted polygon vertex train-fit repair before rerunning train30.

## 117. S279-S283 COMSOL V3 polygon train30 fit repair

S279-S283 repairs the train30 fit gate without changing model structure or runner code. S280 confirms the S275 failure is not target/rasterizer misalignment: sample `21` has target/pred area `63` / `60` but FP/FN pixels `18` / `21`, showing boundary-position error on a small polygon. The failure is broad across hard-case types, so the stage first tests optimization rather than hard-case-specific routing or a new loss.

S282 runs `longer_train30`, the same configuration as S275 with `steps=20000`. It passes the train-fit gate with train mean/min IoU `0.935101` / `0.802920`, presence/type accuracy `1.000000` / `1.000000`, and vertex MAE `5.560893e-05`. All train hard-case groups clear mean IoU `0.75`; the weakest is `geometry_or_type_interaction` at `0.868425`.

By stop-on-pass, larger capacity and area/edge auxiliary runs are skipped. This stage does not promote a polygon inverse candidate and does not replace the S185/S181 center-bin branch candidate. Val/test remain weak observation metrics; the next stage should diagnose held-out generalization before any multi-seed validation.

## 118. S284-S288 COMSOL V3 polygon generalization failure diagnostics

S284-S288 diagnoses held-out failure after train30 fit is repaired. The stage does not run training, does not modify the model or runner, and does not expand data. It adds a read-only generalization diagnostic script that joins S254 raw/converted polygon data with S282 prediction exports.

The coarse split design is not obviously broken. Hard-case counts match design, true rotated rates are train/val/test `0.700` / `0.700` / `0.800`, true multi-component rates are `0.233` / `0.300` / `0.300`, and signal std stays same-scale at `2.124018e-06` / `2.266323e-06` / `1.900533e-06`. The main split caveat is sparse x coverage: test `center_x` mean is `0.008342`, right-shifted relative to train `-0.001439`.

Prediction failure is broad and vertex/shape dominated. Val/test have `4/10` and `6/10` zero-IoU samples, vertex MAE is two orders above train, and held-out predictions show signed-area flips plus occasional out-of-grid vertices. S288 therefore recommends output-shape / vertex-parameterization repair or controlled resplit diagnostics before any multi-seed polygon candidate validation.

## 119. S289-S293 COMSOL V3 center-anchored polygon inverse route

S289-S293 changes the polygon inverse representation from absolute normalized vertices to center-bin localization plus local grid-cell vertices. The target builder adds `center_x_bin_targets`, `center_y_bin_targets`, `center_offset_targets`, and `local_vertices_grid`; center/local decode preserves polygon oracle train/val/test IoU `1.000000` / `1.000000` / `1.000000`.

The new independent runner passes the staged gates. One-sample IoU is `1.000000`; five-sample mean/min IoU is `0.991549` / `0.957746`; train30 mean/min IoU is `0.989276` / `0.857143`, with presence/type and center x/y bin accuracy all `1.000000` on train.

Held-out performance is not solved. Val/test mean IoU is `0.072402` / `0.084416`, and zero-IoU counts are `8/10` and `8/10`. The representation removes the prior signed-area flip / out-of-grid pathology in this run, but the remaining bottleneck is held-out center-bin and local-shape generalization. The S185/S181 center-bin candidate and S282 absolute-vertex polygon reference remain unchanged.

## 120. S294-S297 COMSOL V3 center-anchored polygon held-out diagnostics

S294-S297 diagnoses the remaining center-anchored held-out failure without new training, runner changes, model changes, multi-seed validation, or new COMSOL data. The diagnostic joins S292 prediction exports with S290 center-anchored targets and S254 polygon metadata.

S295 shows the immediate failure mechanism is center-bin localization, especially y-bin. Val/test zero-IoU samples are `16/20`; all `16/16` zero-IoU samples have at least one center-bin error and all `16/16` have a y-bin error, while x-bin errors affect `8/16`. Correct-bin held-out components have lower local vertex grid MAE (`0.867101`) than wrong-bin components (`2.487230`), so local shape is not the first bottleneck.

S296 shows this is also a coverage problem. Held-out data contains `19` component center bins not covered by train; `15/16` zero-IoU samples touch at least one uncovered bin, and zero-IoU samples have higher nearest-train center-bin distance than nonzero samples (`1.468750` vs `0.250000`). S297 therefore recommends a matched-coverage resplit gate on the existing polygon V3 pack before adding model complexity, more steps, multi-seed validation, or larger COMSOL data.

## 121. S298-S302 COMSOL V3 polygon matched-coverage resplit gate

S298-S302 tests that recommendation without generating COMSOL data, changing the model, changing the runner, or entering multi-seed validation. S299 creates a new diagnostic split from the existing 50 polygon V3 samples. Hard-case counts remain at train `10/5/7/5/3`, val `3/2/2/2/1`, and test `3/2/2/2/1`; all `20/20` held-out samples have component bins within train center-bin distance `<=1`, though exact same-bin coverage is only `4/20`.

S300 uses the unchanged center-anchored polygon runner with the same `steps=20000`, `seed=1` setting. Train fit remains strong: mean/min IoU is `0.995598` / `0.969697`, and train presence/type/x-bin/y-bin accuracy are all `1.000000`. Held-out performance does not improve: val/test mean IoU is `0.037245` / `0.072368`, versus the original `0.072402` / `0.084416`, and zero-IoU counts are `8/10` / `9/10`.

S301 shows the matched split still fails through center-bin localization, especially y-bin. Held-out zero-IoU is `17/20`; all `17/17` zero-IoU samples have y-bin errors, and `9/17` have x-bin errors. S302 therefore rules out distance-1 coverage gap as a sufficient explanation and recommends center-anchored y-bin localization repair before multi-seed validation, extra steps, larger model changes, or larger COMSOL data.

## 122. S303-S307 COMSOL V3 center-anchored y-bin localization repair

S303-S307 tests a narrow y-bin repair without changing model structure, target schema, steps, data, or the S185/S181 branch candidate. S304 shows the matched-split reference has train y-bin accuracy `1.000000`, but val/test y-bin accuracy only `0.230769` / `0.083333`; held-out y-bin errors include `6` adjacent errors and `15` distance-`>=2` errors.

S305 adds default-off y-bin extra loss support to the center-anchored runner: `neighbor_soft_ce` and `distance_soft_ce` are added on top of the unchanged hard center-bin CE when explicitly enabled. S306 exactly reproduces the S300 reference. `neighbor_soft_y` partially improves y-bin localization and reduces zero-IoU from `8/10` / `9/10` to `7/10` / `8/10`, but it does not improve both val/test IoU over reference; `distance_soft_y` is worse. S307 therefore fails the gate and recommends local-shape conditioning / bounded local output next, not more y-loss tuning.

## 123. S308-S312 COMSOL V3 center-anchored bounded local output repair

S308-S312 tests bounded local vertex output without changing the center-bin path, model structure, target schema, training steps, matched split, or COMSOL data. S309 shows train/val/test local-shape target ranges are same-scale, so the gate is aimed at predicted local-shape stability rather than target-range mismatch.

S310 adds default-off bounded output support to `train_comsol_center_anchored_polygon_inverse.py`: `--local-shape-output-mode raw|bounded_tanh`, `--local-shape-bound-mode fixed_grid|train_stats`, fixed bounds `[24,8]`, and train-only stats bounds with margin `1.25`. Raw mode preserves previous behavior.

S311 reproduces the reference exactly with val/test IoU `0.037245` / `0.072368`. The bounded variants preserve train fit but fail held-out acceptance: `bounded_local_fixed_grid` reaches val/test IoU `0.024490` / `0.060554`, and `bounded_local_train_stats` reaches `0.029174` / `0.067532`. Both keep signed-area flips and out-of-grid vertices at `0`, and saturation remains `0.0`. S312 therefore recommends local-shape conditioning rather than more bound sweeps, y-loss tuning, multi-seed validation, or new COMSOL data.

## 124. S313-S317 COMSOL V3 center-anchored local-shape conditioning repair

S313-S317 tests default-off local-shape conditioning without changing the center-bin path, target schema, loss weights, matched split, training steps, COMSOL data, or the S185/S181 branch candidate. S314 adds `--local-shape-conditioning-mode none|center_bin|center_bin_slot|center_bin_slot_type`; `none` preserves the previous behavior, while conditioned modes feed the local vertex head with shared latent plus detached predicted center-bin context, center offset, optional slot embedding, and optional predicted type context.

S315 smoke tests pass for both default and conditioned model/runner paths. S316 first reproduces the S311 reference exactly: train mean/min IoU `0.995598` / `0.969697`, val/test IoU `0.037245` / `0.072368`, and zero-IoU `8/10` / `9/10`.

The first conditioned variant, `conditioning_center_bin`, improves train fit and local train MAE but fails held-out acceptance: val/test IoU are `0.027215` / `0.067059`, below the reference, with val zero-IoU worsening to `9/10`. Slot and type variants are skipped by stop condition. S317 therefore recommends a joint center-bin/local-shape repair rather than more simple local-conditioning variants, y-loss tuning, bound sweeps, multi-seed validation, or new COMSOL data.

## 125. S318-S322 COMSOL V3 center-anchored joint center/local diagnosis

S318-S322 starts from the failed S313-S317 local-conditioning probe and first runs an offline oracle / teacher-forced ablation, not a new training sweep. S319 confirms the ablation path is aligned: `pred_all` reproduces the exported metrics, and `gt_center_bin_offset_local` reaches val/test IoU `1.000000` / `1.000000`.

The causal result is clear. On the matched-split reference, `gt_center_bin_offset` raises val/test IoU from `0.037245` / `0.072368` to `0.450778` / `0.438502` and removes zero-IoU samples; `gt_local` only reaches `0.058471` / `0.095985`. On `conditioning_center_bin`, `gt_center_bin_offset` reaches `0.495734` / `0.574024`, while `gt_local` stays at `0.034444` / `0.055102`. Held-out failure is therefore dominated by center decode, not by local vertices alone.

Because that center-main-cause gate passed, S320-S321 tests one default-off joint repair, `joint_center_shape_mode=soft_center_scheduled`. The reference reproduces S316 exactly, but the repair fails: train mean/min IoU is `0.977544` / `0.818182`, val/test IoU is `0.000000` / `0.090810`, and val/test zero-IoU is `10/10` / `7/10`. S322 therefore stops the route at diagnosis: the next step should be an explicit center-local coupling design, not more ad hoc variants, multi-seed validation, additional steps, model scaling, or new COMSOL data.

## 126. S323-S327 COMSOL V3 center-anchored decoded-center coupling repair

S323-S327 tests the smallest loss-side repair implied by S319: differentiable decoded-center consistency. The runner keeps hard argmax center decode for official mask metrics and prediction export, but adds default-off `--center-consistency-mode none|soft_decoded_center|soft_decoded_vertex` and `--lambda-center-consistency`.

S324 confirms the matched-split reference still has severe held-out center decode error. The same-run reference has val/test IoU `0.037245` / `0.072368`, y-bin accuracy `0.230769` / `0.083333`, and hard decoded center L2 grid error `21.748188` / `15.275539`. S326 reproduces that reference exactly before testing any repair.

The first repair, `soft_decoded_center_consistency`, does not pass. It improves train center error but drops train mean/min IoU to `0.983633` / `0.857143`, collapses val IoU to `0.000000`, and lowers test IoU to `0.034211`. By stop condition, `soft_decoded_vertex_consistency` is skipped. S327 therefore recommends a structural component-query center/shape head or equivalent shared component representation next, not more loss-weight tuning, y-loss, local-conditioning, bound sweeps, teacher-forcing variants, multi-seed validation, extra steps, or new COMSOL data.

## 127. S328-S332 COMSOL V3 component-query center/shape head

S328-S332 implements an independent component-query polygon route without replacing S185/S181, the absolute polygon runner, or the center-anchored runner. The new model uses a shared MLP encoder plus three learned component queries. Each query latent jointly predicts presence, type, center x/y bins, center offset, and local vertices.

S329 smoke tests pass for the model and runner, including finite loss/backward, query/head gradients, tempfile training, metrics/history/summary writing, prediction export, and no checkpoint/weight/`.npy` output. The old center-anchored runner default path is left unchanged.

The route stops at S330. The one-sample overfit gate on train sample `0` reaches presence/type/x-bin/y-bin accuracy `1.000000` and decoded vertex MAE `5.918177e-06`, but hard polygon IoU is `0.974227`, below the required `>=0.99`; pred/target area is `194` / `189`. By stop condition, the 5-sample gate, same-run reference, and train30 quick gate are skipped. The next step should diagnose component-query one-sample raster sensitivity before scaling the route.

## 128. S333-S335 COMSOL V3 component-query 1-sample raster sensitivity

S333-S335 diagnoses the S330 one-sample failure without new training, runner changes, model changes, 5-sample, train30, multi-seed validation, or new COMSOL data. The offline diagnostic reconstructs the S330 predicted polygon and exactly reproduces the exported IoU `0.974227` and pred/target area `194` / `189`.

The error is small and one-sided: `5` false-positive pixels, `0` false-negative pixels, and max vertex error only `0.039043` grid cells. Area and edge corrections alone are not enough: area-scaled and edge-scaled variants reach only `0.979275`. The decisive ablation is center/centroid: `gt_center + pred_local_vertices` reaches IoU `1.000000`, while `pred_center + gt_local_vertices` remains `0.979275`; centroid alignment also reaches `1.000000`.

S335 therefore keeps the 5-sample gate blocked. The next step should be a 1-sample component-query precision repair focused on center / centroid alignment, not train30, held-out generalization, multi-seed validation, or new COMSOL data.

## 129. S336-S340 COMSOL V3 component-query center precision repair

S336-S340 tests that 1-sample precision repair with default-off decoded-center and polygon-centroid auxiliary losses. S337 adds `--lambda-decoded-center-aux`, `--lambda-polygon-centroid-aux`, and `--center-centroid-aux-smoothl1-beta`; all defaults preserve old runner behavior.

The same-run reference reproduces S330 exactly: IoU `0.974226804`, pred/target area `194 / 189`, and presence/type/x-bin/y-bin accuracy `1.000000`. `decoded_center_aux_small` improves IoU to `0.984126984`, but it changes the miss from `5 / 0` FP/FN to `0 / 3` FP/FN and area `186 / 189`; `polygon_centroid_aux_small` is worse at IoU `0.963917526`. The conditional combined run is skipped.

The 1-sample gate remains failed, so 5-sample and train30 stay blocked. The next component-query step should be a more targeted boundary/center-local precision repair, not more center/centroid lambda tuning, multi-seed validation, or new COMSOL data.

## 130. S341-S345 COMSOL V3 component-query boundary precision repair

S341-S345 tests a narrower 1-sample repair after S336-S340 shows center/centroid aux is not sufficient. The only runner change is exposing the existing default-off `--lambda-area-aux` in the component-query runner; the model and default behavior remain unchanged.

The best run is `center_aux_half`: IoU `0.989528796`, pred/target area `191 / 189`, and FP/FN `2 / 0`. It improves over current reference IoU `0.974226804` and reduces symmetric diff from `5` to `2`, but still misses the explicit `>=0.99` acceptance threshold. Adding tiny area aux worsens IoU to `0.979166667`.

The stage therefore keeps 5-sample and train30 blocked. The next route should be targeted boundary-aware repair or an explicit gate review, not multi-seed, new COMSOL data, or a broad loss sweep.
