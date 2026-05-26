# S31 跨分辨率报告图表

## 数据来源

本步骤只读取 `experiments/dual_network/S30_cross_resolution_report/aggregated_metrics.csv`，没有运行新训练，也没有重新计算单样本结果。

## 生成文件

- `defect_iou_by_resolution.svg`：展示 20x10、40x20、80x40 下 baseline 与 BCE/default 配置的 avg `defect_iou`。
- `defect_area_pred_by_resolution.svg`：展示三种分辨率下 baseline 与 BCE/default 配置的 avg `defect_area_pred`。
- `mu_error_by_resolution.svg`：展示三种分辨率下 baseline 与 BCE/default 配置的 avg `mu_mae`；`mu_mse` 保留在 `aggregated_metrics.csv` 和顶层报告表格中。

## 主要趋势

- 20x10、40x20、80x40 三个分辨率下，BCE/default 配置的 avg `defect_iou` 均明显高于 baseline。
- baseline 的 avg `defect_area_pred` 在所有分辨率下都偏大，尤其 80x40 下低 `mu` 区域扩散更明显。
- BCE/default 配置显著压低 false positives，并同步降低 `mu_mae`。
- 高分辨率下仍需要匹配 `mask-prior-temperature` 和 `lambda-mask-bce-prior`，但当前 80x40 `temp25_lambda3` 已明显优于 baseline。

## 边界说明

这些图表只用于展示 S30 已聚合结果，不代表新实验。当前结论仍是：BCE mask prior 是半监督 / 诊断上界，因为它使用 `mu_label < 500` 的 mask 信息；不能表述为纯无监督 weak-form 反演成功。
