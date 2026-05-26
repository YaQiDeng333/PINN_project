# S116 COMSOL parametric route summary

## 为什么切换到 parametric route

V2 dense conditional mask runner 已经在多轮诊断中出现全背景、全前景和 localization 不足。S111 已规定后续不再直接跑 full V2 long-run 搜索。V2 数据同时提供 component-level `defect_params`，因此本阶段测试更低维的 geometry parameter inverse route。

## S113 targets

S113 成功从 V2 `defect_params.csv` 中构造 train / val / test parametric targets：

- `max_components=3`
- 所有样本均有 3 个 component，无截断。
- `type_vocab=rectangular_notch, rotated_rect`
- continuous schema 为 `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- train / val / test 的 type vocab 和参数范围一致或高度重叠。

## S114 model

S114 创建 `ParametricInverseNet` skeleton：

- 输入 flattened signal `[B,600]`
- 输出 `presence_logits`、`type_logits` 和 normalized continuous geometry parameters
- smoke test 验证 forward / backward shape 正常。

## S115 probe

S115 使用 V2 train=100 / val=20 / test=20 运行首个 parametric inverse probe。主要结果：

- train / val / test presence accuracy: `1.0` / `1.0` / `1.0`
- train / val / test type accuracy: `1.0` / `0.65` / `0.6667`
- val / test center MAE: `1.485341e-03` / `1.285206e-03`
- val / test rotation MAE: `7.731843` / `7.740396` degree
- val / test rasterized param mask IoU: `0.369908` / `0.424462`

## 是否比 dense mask route 更有希望

当前结果支持继续推进 parametric route。它没有出现 dense V2 runner 的全背景 / 全前景塌缩，并且通过非可微 rasterization 得到的 val/test mask IoU 已有非平凡信号。

这仍不是最终结论：parametric route 的 mask IoU 来自近似 rasterization，且训练使用了 full V2 train split；后续仍需 quick gate、稳定性验证和更严格的 val selection。

## 当前瓶颈

1. `rotation_angle` 泛化误差仍偏高。
2. `component_type` val/test accuracy 只有约 `0.65`。
3. rasterization 目前是近似 rectangle / rotated rectangle，尚未覆盖更复杂 geometry。
4. 当前 encoder 仍是简单 MLP，可能不足以提取多高度 Bz 的局部结构。

## 下一步建议

- 如果继续 parametric route，先补充更严格的 Gate 1 / Gate 2 验证。
- 实现更完整的 rasterized mask IoU report，包括按 defect type / rotation / component distance 分组。
- 如果 center/type 较好但尺寸或角度差，增强 continuous loss 的分组权重或角度表示。
- 如果后续参数预测不稳定，再考虑更强 signal encoder，例如 1D CNN / attention encoder。
