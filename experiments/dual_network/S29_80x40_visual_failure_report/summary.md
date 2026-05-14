# S29 80x40 可视化失败诊断

## 数据来源

S29 只读取已有 S28 输出，没有重新训练。

- 来源目录：`experiments/dual_network/S28_80x40_50sample_default_validation/`
- baseline 指标：`baseline/metrics.csv`
- 候选配置指标：`temp25_lambda3/metrics.csv`
- 代表性图像来自：`temp25_lambda3/sample_*/mu_pred_vs_label.png`

## S28 总体结果

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.051006e-01 | 1.242440e+03 | 3.443720e+05 | 3.814918e+02 |
| temp25_lambda3 | 8.925113e-01 | 1.328200e+02 | 4.572207e+04 | 1.832775e+02 |

`temp25_lambda3` 在 50/50 个样本上 IoU 都优于 baseline，并显著降低 `defect_area_pred`、`mu_mse` 和 `mu_mae`。

## 代表性图像集合

已复制 15 张代表性 PNG 到 `figures/`：

- `figures/sample_02_mu_pred_vs_label.png`
- `figures/sample_29_mu_pred_vs_label.png`
- `figures/sample_47_mu_pred_vs_label.png`
- `figures/sample_45_mu_pred_vs_label.png`
- `figures/sample_48_mu_pred_vs_label.png`
- `figures/sample_41_mu_pred_vs_label.png`
- `figures/sample_49_mu_pred_vs_label.png`
- `figures/sample_21_mu_pred_vs_label.png`
- `figures/sample_03_mu_pred_vs_label.png`
- `figures/sample_37_mu_pred_vs_label.png`
- `figures/sample_11_mu_pred_vs_label.png`
- `figures/sample_25_mu_pred_vs_label.png`
- `figures/sample_44_mu_pred_vs_label.png`
- `figures/sample_06_mu_pred_vs_label.png`
- `figures/sample_18_mu_pred_vs_label.png`

## 代表性成功样本：IoU 最高

| sample | baseline_iou | temp25_iou | improvement | area_pred / area_label | centroid_offset | figure |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 8.422665e-02 | 1.000000e+00 | 9.157734e-01 | 55 / 55 | 0.000000e+00 | `figures/sample_02_mu_pred_vs_label.png` |
| 29 | 1.117705e-01 | 1.000000e+00 | 8.882295e-01 | 115 / 115 | 0.000000e+00 | `figures/sample_29_mu_pred_vs_label.png` |
| 47 | 4.613734e-02 | 1.000000e+00 | 9.538627e-01 | 43 / 43 | 0.000000e+00 | `figures/sample_47_mu_pred_vs_label.png` |

这些样本中，`temp25_lambda3` 的 `defect_area_pred` 与 `defect_area_label` 完全一致或接近一致，centroid 偏移为 0 或近似 0。

## 明显失败或弱样本：IoU 最低

| sample | baseline_iou | temp25_iou | improvement | area_pred / area_label | centroid_offset | figure |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 45 | 5.772812e-02 | 6.488550e-01 | 5.911269e-01 | 122 / 94 | 4.152222e-01 | `figures/sample_45_mu_pred_vs_label.png` |
| 48 | 5.030488e-02 | 6.966292e-01 | 6.463243e-01 | 77 / 74 | 6.153639e-02 | `figures/sample_48_mu_pred_vs_label.png` |
| 41 | 8.925144e-02 | 7.068965e-01 | 6.176451e-01 | 99 / 99 | 2.179779e-02 | `figures/sample_41_mu_pred_vs_label.png` |
| 49 | 5.229456e-02 | 7.413793e-01 | 6.890848e-01 | 51 / 50 | 6.777946e-02 | `figures/sample_49_mu_pred_vs_label.png` |
| 21 | 1.105372e-01 | 7.518519e-01 | 6.413147e-01 | 259 / 214 | 2.972114e-01 | `figures/sample_21_mu_pred_vs_label.png` |

这些样本仍明显优于 baseline，但相比成功样本，弱点集中在形状细节、面积偏差或 centroid 偏移。

## 面积误差最大的样本

| sample | temp25_iou | area_pred / area_label | centroid_offset | figure |
| ---: | ---: | ---: | ---: | --- |
| 21 | 7.518519e-01 | 259 / 214 | 2.972114e-01 | `figures/sample_21_mu_pred_vs_label.png` |
| 45 | 6.488550e-01 | 122 / 94 | 4.152222e-01 | `figures/sample_45_mu_pred_vs_label.png` |
| 3 | 9.109589e-01 | 136 / 143 | 3.706090e-02 | `figures/sample_03_mu_pred_vs_label.png` |
| 37 | 9.125000e-01 | 73 / 80 | 2.411117e-02 | `figures/sample_37_mu_pred_vs_label.png` |
| 11 | 9.130435e-01 | 85 / 91 | 1.373796e-02 | `figures/sample_11_mu_pred_vs_label.png` |

面积误差最大的样本并不都表现为低 IoU，说明面积误差需要结合位置和形状一起判断。

## centroid 偏移最大的样本

| sample | temp25_iou | area_pred / area_label | centroid_offset | figure |
| ---: | ---: | ---: | ---: | --- |
| 45 | 6.488550e-01 | 122 / 94 | 4.152222e-01 | `figures/sample_45_mu_pred_vs_label.png` |
| 21 | 7.518519e-01 | 259 / 214 | 2.972114e-01 | `figures/sample_21_mu_pred_vs_label.png` |
| 25 | 8.076923e-01 | 94 / 94 | 9.225532e-02 | `figures/sample_25_mu_pred_vs_label.png` |
| 44 | 8.956522e-01 | 110 / 108 | 7.751736e-02 | `figures/sample_44_mu_pred_vs_label.png` |
| 6 | 8.557692e-01 | 97 / 96 | 6.792095e-02 | `figures/sample_06_mu_pred_vs_label.png` |

centroid 偏移最大的样本中，sample 45 和 sample 21 也是低 IoU / 面积误差样本，说明它们是优先复查对象。

## baseline 到候选配置改善最大的样本

| sample | baseline_iou | temp25_iou | improvement | area_pred / area_label | centroid_offset | figure |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 47 | 4.613734e-02 | 1.000000e+00 | 9.538627e-01 | 43 / 43 | 0.000000e+00 | `figures/sample_47_mu_pred_vs_label.png` |
| 18 | 6.547931e-02 | 9.841270e-01 | 9.186477e-01 | 125 / 125 | 2.279379e-02 | `figures/sample_18_mu_pred_vs_label.png` |
| 2 | 8.422665e-02 | 1.000000e+00 | 9.157734e-01 | 55 / 55 | 0.000000e+00 | `figures/sample_02_mu_pred_vs_label.png` |
| 26 | 5.828517e-02 | 9.593496e-01 | 9.010644e-01 | 120 / 121 | 2.375141e-02 | `` |
| 20 | 6.072289e-02 | 9.603174e-01 | 8.995945e-01 | 121 / 126 | 1.712973e-02 | `` |

这些样本说明 `temp25_lambda3` 能有效抑制 baseline 的大面积 false positives，并恢复接近真实的缺陷区域。

## baseline 到候选配置改善最小的样本

| sample | baseline_iou | temp25_iou | improvement | area_pred / area_label | centroid_offset | figure |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 45 | 5.772812e-02 | 6.488550e-01 | 5.911269e-01 | 122 / 94 | 4.152222e-01 | `figures/sample_45_mu_pred_vs_label.png` |
| 41 | 8.925144e-02 | 7.068965e-01 | 6.176451e-01 | 99 / 99 | 2.179779e-02 | `figures/sample_41_mu_pred_vs_label.png` |
| 21 | 1.105372e-01 | 7.518519e-01 | 6.413147e-01 | 259 / 214 | 2.972114e-01 | `figures/sample_21_mu_pred_vs_label.png` |
| 48 | 5.030488e-02 | 6.966292e-01 | 6.463243e-01 | 77 / 74 | 6.153639e-02 | `figures/sample_48_mu_pred_vs_label.png` |
| 49 | 5.229456e-02 | 7.413793e-01 | 6.890848e-01 | 51 / 50 | 6.777946e-02 | `figures/sample_49_mu_pred_vs_label.png` |

改善最小的样本仍然比 baseline 好很多，但弱点集中在局部形状、centroid 和边界情况。

## 主要失败类型判断

- `temp25_lambda3` 在 `80x40` 下整体有效，50/50 个样本的 IoU 都高于 baseline。
- 主要弱样本为 sample 45、48、41、49、21，其中 sample 45 和 48 低于 IoU `0.7`。
- 主导失败模式不是全域低 `mu` 塌陷；S28 已经明显压制 baseline 的面积扩张。
- 剩余弱样本更像是形状细节不匹配、边界 / 窄缺陷样本、centroid 偏移或局部几何误差。
- 面积误差和 centroid 偏移异常样本应优先做人工可视化复查。

## 当前结论

- `temp25_lambda3` 可以作为当前 `80x40` 综合默认候选。
- S29 支持从参数搜索转向结果整理和失败样本复查。
- 这些结果仍是半监督 / 诊断上界，因为 BCE 和 mask prior 使用 `mu_label < 500`；不能表述为无监督 weak-form 反演成功。

## 下一步建议

- 优先做最终报告 / 论文式结果整理。
- 不建议继续大规模扫描 `test_radius`、`center_mode` 或 `area prior`。
- 如果要改善弱样本，应先分析缺陷形状、边界位置和局部几何，而不是盲目调整 loss 权重。
