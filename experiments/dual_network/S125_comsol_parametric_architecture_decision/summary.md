# S125 COMSOL parametric architecture decision

## S121 error decomposition

S121 显示 S115 train 已接近 oracle，但 val/test oracle gap 明显：

- S115 val / test oracle gap = `3.533803e-01` / `2.921214e-01`。
- S119 val / test oracle gap = `3.975235e-01` / `3.280746e-01`。

主要 held-out 误差来自 type accuracy 约 `0.62` 到 `0.68`、rotation MAE 约 `7` 到 `8` degree，以及 geometry prediction 到 oracle upper bound 之间的泛化 gap。presence 已基本解决。

## S122 新增能力

S122 为 parametric inverse route 增加：

- `encoder_type=mlp|cnn1d|cnn1d_attention`
- `head_mode=shared|component_specific`
- continuous group loss weights: `lambda_center`、`lambda_axis`、`lambda_depth`、`lambda_rotation`

默认 `encoder_type=mlp`、`head_mode=shared` 保持旧行为。

## S123 quick gate

S123 比较四组：

- `raw_mlp_shared_reference`
- `raw_mlp_component_specific`
- `raw_cnn_component_specific`
- `raw_attention_component_specific`

其中 `raw_cnn_component_specific` 是 S123 内部最佳配置，val / test mask IoU = `3.662237e-01` / `3.892232e-01`。它改善了 val type accuracy 和 rotation MAE，但没有超过 S115 raw baseline。

## S124 full probe

S124 对 `raw_cnn_component_specific` 执行 6000-step full probe。结果 val / test mask IoU = `3.133465e-01` / `3.153288e-01`，低于 S115 与 S123 quick result。

## 当前最佳配置

当前整体最佳仍是 S115 raw parametric MLP baseline：

- val mask IoU = `3.699078e-01`
- test mask IoU = `4.244624e-01`

S123/S124 的 component-specific heads 与 CNN/attention encoder 没有形成新的默认配置。

## 与 dense route 对比

parametric route 仍继续成立，因为它避免了 dense V2 runner 的全背景 / 全前景塌缩，并且 S117 oracle gate 显示 target+rasterizer 有约 `0.72` IoU 上限。但当前 architecture probe 没有证明更强 encoder 或 component-specific heads 已解决 held-out geometry generalization。

## 下一步建议

- 不继续拉长当前 CNN/component-specific 配置。
- 优先增加 per-sample prediction export 与 grouped diagnostics，按 type、rotation bin、component slot 和 target area 拆解误差。
- 如果要继续 architecture，优先考虑更明确的 slot decoder / set prediction，而不是简单 independent heads。
- 如果要进入 forward consistency / differentiable rasterization，应先固定 rasterizer semantics，并把 S117 oracle 作为上限参考。
- 如果 type 改善但 mask IoU 不改善，应加强 rasterization / mask refinement 或 geometry-to-mask calibration。
