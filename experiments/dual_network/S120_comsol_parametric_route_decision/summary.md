# S120 COMSOL parametric route decision

## S117 oracle rasterization 上限

GT parametric targets 经 rasterizer 重建 target masks 的平均 oracle IoU：

- train: `7.229967e-01`
- val: `7.232882e-01`
- test: `7.165838e-01`

三者均高于 `0.70` gate，说明当前 parametric target + rasterizer 语义有足够上限，可以继续 parametric route。

## S118 target refinement

S118 引入：

- `rotation_sin` / `rotation_cos`
- train-stat continuous normalization
- raw targets 保留用于真实单位 metrics 和 rasterization
- `--type-class-weighting inverse_freq`

这些修改使 target artifact 的语义更清楚，但不保证当前 MLP 一定泛化更好。

## S119 training result

`refined_mlp` 完成 3000 steps。相比 S115：

- rotation MAE 没有稳定改善；
- type accuracy 没有稳定改善；
- val/test mask IoU 下降到 `0.325765` / `0.388509`；
- 因此跳过 `refined_mlp_longer`。

## 当前 parametric route 是否成立

成立，但需要换下一步重点。

理由：
- S117 oracle IoU 证明 target+rasterizer 上限足够；
- S115 原始 parametric probe 已经优于近期 dense route failure；
- S119 refinement 没有提升，说明瓶颈更可能是 encoder / head / loss 分解，而不是单纯 angle encoding 或 normalization。

## 与 dense route 对比

dense V2 route 在 S93 / S97 / S101 / S109 中反复出现全背景、全前景或 localization 不足。parametric route 至少能稳定预测 presence，并给出非零、结构化的 held-out geometry signal。

## 下一步建议

- 继续 parametric route，但不要继续只拉长当前 MLP。
- 下一步优先：
  - 1D CNN / attention signal encoder；
  - component-specific head 或 slot-wise decoder；
  - type / rotation 分离 loss；
  - forward consistency 或 differentiable rasterization；
  - 按 type / rotation / component distance 分组评估 mask IoU。
