# S127 COMSOL parametric grouped diagnostics

## 目的

S127 基于 S126 的 per-sample / per-component predictions，对 S115 raw MLP baseline 做 grouped diagnostics，定位 type、rotation、slot、area 和 oracle gap 对 mask IoU 的影响。

## 主要结论

- `rectangular_notch` 与 `rotated_rect` 在全部 split 合并后的 type accuracy 接近，分别为 `9.095238e-01` 与 `8.952381e-01`；但这是 train 主导后的 aggregate，held-out val/test 仍是 S115 中约 `0.65` / `0.6667`。
- component slot 1 的 type accuracy 较低，为 `8.571429e-01`；slot 0 / slot 2 分别为 `9.285714e-01` / `9.214286e-01`。
- rotation error bin 与 mask IoU 明显相关：`0-5` degree bin 的 mean mask IoU 为 `6.380047e-01`，而 `5-10` / `10-20` / `>30` bins 的 mean mask IoU 约为 `4.165229e-01` / `3.995313e-01` / `4.356924e-01`。
- worst samples 中存在较大的 oracle gap，例如 test sample `10` 的 pred IoU = `3.188098e-03`，oracle IoU = `7.455231e-01`，说明低 IoU 主要来自 model prediction error，而不是 rasterizer upper bound。

## 对 set matching 的支持程度

S127 显示 slot 1 更弱，且部分低 IoU 样本并非单纯 type sequence 错误；这支持测试 fixed-order slot regression 是否存在 order ambiguity。但 grouped diagnostics 也显示 rotation / geometry error 与 oracle gap 更直接相关，因此 permutation matching 只是一个必要诊断，不应被预设为最终解决方案。

## 自评

- grouped CSV 与 worst samples 已生成。
- 诊断结果足以支持 S128/S129 的 set-matching probe。
- 未发现需要再次 Claude review 的 schema 风险。
