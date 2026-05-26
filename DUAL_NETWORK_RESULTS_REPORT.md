# DUAL_NETWORK_RESULTS_REPORT

## 1. 支线结果一句话结论

双网络 weak-form 框架已经工程跑通；纯 weak-form / area / Dice baseline 定位能力不足；加入 BCE mask prior 后，在 20x10、40x20、80x40 小规模样本上均显著优于 baseline，但该结果属于半监督 / 诊断上界，不是无监督 weak-form 反演成功。

## 2. 方法配置简述

本支线使用 `phi-Net` / `mu-Net` 双网络结构：`phi-Net` 重构磁标量势场，`mu-Net` 输出材料磁导率分布。`mu-Net` 阶段使用来自 `div(mu grad phi)=0` 的 weak-form material update，并通过 compact-support test gradients 构造弱形式残差。

实验中比较两类配置：

- `baseline`：weak-form + TV + area prior + soft Dice mask prior，不使用 BCE mask prior。
- `bce` / `temp25_lambda1` / `temp25_lambda3`：在 baseline 基础上加入 BCE mask prior。BCE 使用 `mu_label < 500` 构造 label mask，因此属于半监督 / 诊断上界。

## 3. 跨分辨率结果表

| stage | resolution | config | samples | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| S19 | 20x10 | baseline | 50 | 1.101394e-01 | 6.504000e+01 | 2.871279e+05 | 3.216540e+02 |
| S19 | 20x10 | bce | 50 | 8.399348e-01 | 9.120000e+00 | 1.935842e+04 | 6.434378e+01 |
| S24 | 40x20 | baseline | 50 | 1.426606e-01 | 2.353600e+02 | 2.503184e+05 | 2.932973e+02 |
| S24 | 40x20 | temp25_lambda1 | 50 | 9.203000e-01 | 3.340000e+01 | 3.598542e+04 | 1.498937e+02 |
| S28 | 80x40 | baseline | 50 | 1.051006e-01 | 1.242440e+03 | 3.443720e+05 | 3.814918e+02 |
| S28 | 80x40 | temp25_lambda3 | 50 | 8.925113e-01 | 1.328200e+02 | 4.572207e+04 | 1.832775e+02 |

## 4. 结果解释

- BCE mask prior 在所有分辨率下都显著改善 `defect_iou`。
- baseline 的 `defect_area_pred` 通常明显偏大，说明 weak-form / area / Dice 仍容易产生低 `mu` 扩散和 false positives。
- BCE 能显著压制 false positives，并同步降低 `mu_mse` / `mu_mae`。
- 高分辨率下需要适配 `mask-prior-temperature` 和 `lambda-mask-bce-prior`：40x20 的 IoU 优先候选是 `temp25_lambda1`，80x40 的综合候选是 `temp25_lambda3`。
- S28 / S29 显示 80x40 下结果整体可用，但弱样本仍和形状细节、边界/窄缺陷、centroid 偏移和局部几何误差有关。

## 5. 当前边界

- 当前不能声称纯无监督 weak-form 反演成功。
- BCE mask prior 使用 `mu_label` mask。
- `label-informed centers` 是 oracle diagnostic，不是可部署方法。
- 当前支线不替代 `main`。
- 当前最适合表述为半监督双网络上界路线。

## 6. 后续建议

继续半监督双网络路线，而不是继续盲目扫描 pure weak-form 参数。

下一阶段建议：

1. 固定 `train_dual_variational.py` runner。
2. 以 40x20 / 80x40 默认候选为基础继续整理结果。
3. 做更系统的可视化与失败样本分类。
4. 再考虑是否写成论文方法补充或支线结果章节。

## 7. 图表索引

这些 SVG 基于 S30 `aggregated_metrics.csv` 生成，未运行新训练。

- [defect_iou_by_resolution.svg](experiments/dual_network/S31_report_figures/defect_iou_by_resolution.svg)
- [defect_area_pred_by_resolution.svg](experiments/dual_network/S31_report_figures/defect_area_pred_by_resolution.svg)
- [mu_error_by_resolution.svg](experiments/dual_network/S31_report_figures/mu_error_by_resolution.svg)
