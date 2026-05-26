# S30 跨分辨率结果报告

## 数据来源

本报告只读取既有实验产物，没有重新训练模型。

- `S19_runner_50sample_bce_validation/`：20x10，50 samples，baseline vs `bce`。
- `S24_40x20_50sample_default_validation/`：40x20，50 samples，baseline vs `temp25_lambda1`。
- `S28_80x40_50sample_default_validation/`：80x40，50 samples，baseline vs `temp25_lambda3`。
- `S29_80x40_visual_failure_report/summary.md`：80x40 可视化失败诊断。

## 聚合指标

| stage | resolution | config | samples | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| S19 | 20x10 | baseline | 50 | 1.101394e-01 | 6.504000e+01 | 2.871279e+05 | 3.216540e+02 |
| S19 | 20x10 | bce | 50 | 8.399348e-01 | 9.120000e+00 | 1.935842e+04 | 6.434378e+01 |
| S24 | 40x20 | baseline | 50 | 1.426606e-01 | 2.353600e+02 | 2.503184e+05 | 2.932973e+02 |
| S24 | 40x20 | temp25_lambda1 | 50 | 9.203000e-01 | 3.340000e+01 | 3.598542e+04 | 1.498937e+02 |
| S28 | 80x40 | baseline | 50 | 1.051006e-01 | 1.242440e+03 | 3.443720e+05 | 3.814918e+02 |
| S28 | 80x40 | temp25_lambda3 | 50 | 8.925113e-01 | 1.328200e+02 | 4.572207e+04 | 1.832775e+02 |

## 跨分辨率观察

- 20x10：`bce` 相比 baseline 将 avg `defect_iou` 从 1.101394e-01 提高到 8.399348e-01，同时显著降低 `defect_area_pred`、`mu_mse` 和 `mu_mae`。
- 40x20：`temp25_lambda1` 相比 baseline 将 avg `defect_iou` 从 1.426606e-01 提高到 9.203000e-01，说明 40x20 下 BCE 半监督上界依然稳定。
- 80x40：`temp25_lambda3` 相比 baseline 将 avg `defect_iou` 从 1.051006e-01 提高到 8.925113e-01，并将平均预测缺陷面积从 1.242440e+03 降到 1.328200e+02。

## S29 失败样本诊断摘要

S29 显示，80x40 `temp25_lambda3` 已经整体抑制 baseline 的低 `mu` 扩散问题。代表性成功样本包括 sample 2、29、47；弱样本主要包括 sample 45、48、41、49、21。弱样本的主要问题不是全域塌陷，而是形状细节、边界/窄缺陷、局部几何误差和少量 centroid 偏移。

## 当前阶段结论

BCE mask prior 在 20x10、40x20、80x40 三个分辨率下都显著优于 baseline。baseline 的 `defect_area_pred` 通常明显偏大，说明当前 pure weak-form / area / Dice 组合仍容易产生低 `mu` 扩散；BCE 能明显压制 false positives 并改善 IoU。

但 BCE 使用 `mu_label < 500` 的 mask 信息，因此这些结果是半监督 / 诊断上界，不是纯无监督 weak-form 反演成功。当前支线最适合表述为半监督双网络路线，而不是 label-free weak-form 已经解决反演问题。

## 图表

本次环境中 matplotlib 不可用，因此跳过 PNG 图表生成。`aggregated_metrics.csv` 已包含绘图所需的全部聚合指标，后续可在有 matplotlib 的环境中复现 `defect_iou_by_resolution`、`defect_area_pred_by_resolution` 和 `mu_error_by_resolution` 图表。
