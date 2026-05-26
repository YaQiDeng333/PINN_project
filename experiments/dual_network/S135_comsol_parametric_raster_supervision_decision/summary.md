# S135 COMSOL parametric raster-supervision decision

## S131 结论

S131 确认 S126-S130 的 set matching 诊断已完成：prediction export 和 grouped diagnostics 可用，但 `permutation_min` 明显低于 fixed-order baseline。因此下一步不继续 loss-side set matching，而是测试 differentiable raster mask supervision。

## S132 / S133 新能力

S132 新增 differentiable soft rasterizer，沿用 S117 hard rasterizer 的 full width / full height axis 语义，并用 sigmoid soft rectangle + probabilistic union 生成 soft mask。

S133 将 raster BCE / Dice loss 接入 `train_comsol_parametric_inverse.py`：

- `--lambda-raster-bce`
- `--lambda-raster-dice`
- `--raster-softness-cells`
- `--raster-target-source`

默认 raster loss 关闭，保持 S115 raw MLP baseline 兼容。

## S134 结果

S134 对比参数-only baseline 与三组 raster-supervised 配置：

- `param_only_reference` val / test mask IoU = `3.699078e-01` / `4.244624e-01`
- `raster_dice1` val / test mask IoU = `3.523885e-01` / `4.385081e-01`
- `raster_bce05_dice1` val / test mask IoU = `3.697576e-01` / `4.096553e-01`
- `raster_dice1_soft2` val / test mask IoU = `3.577784e-01` / `4.088950e-01`

`raster_dice1` 小幅提升 test mask IoU，但 val 下降；因此它不是稳定超过 S115 baseline 的新默认配置。

## 当前最佳配置

当前最佳稳定配置仍是 S115 / S134 `param_only_reference` raw MLP / shared head / fixed-order baseline。

## 是否继续 differentiable rasterization

继续，但不应直接把 S134 raster loss 配置升为默认。S134 说明 raster supervision 有潜在信号，尤其是 test IoU 小幅提升，但当前 loss 权重和训练方式未带来稳定 val/test 改善。

## 下一步建议

- 测试两阶段训练：先 parameter-only 拟合，再短程 raster fine-tune。
- 为 raster loss 加 validation-aware selection，避免 raster objective 牺牲 val。
- 继续探索 forward consistency / differentiable rasterization，但应保留 S115 baseline 作为默认参考。
- 如果 raster loss 继续只改善单侧 split，优先做 regularization 或 grouped validation，而不是继续盲扫权重。

## 自评

- S131-S135 已完成。
- 结论避免过度声称：raster supervision 有信号，但没有稳定超过 S115。
- 下一步仍在 parametric route 内，不回到 dense mask loss 小修。
