# S130 COMSOL parametric set-matching decision

## S126 prediction export

S126 成功为 parametric runner 增加 `--export-predictions`，输出 component-level predictions 和 sample-level mask metrics。新增导出不改变默认 fixed-order 训练结果，S126 复现了 S115 raw MLP baseline：
- val mask IoU = `3.699078e-01`
- test mask IoU = `4.244624e-01`

## S127 grouped diagnostics

S127 显示当前失败不是 presence 问题。主要问题集中在 held-out type / rotation / geometry generalization：
- held-out type accuracy 仍约 `0.65` 到 `0.6667`。
- rotation error bin 与 mask IoU 明显相关，高 rotation error bin 的 mean mask IoU 更低。
- worst samples 有明显 oracle gap，说明模型预测离 oracle upper bound 仍远。
- slot 1 相对更弱，但不足以单独解释全部 held-out gap。

## S128 set-matching support

S128 增加 `component_matching_mode=fixed|permutation_min`。`permutation_min` 对 `max_components=3` 枚举 6 个 permutation，选择最小 component loss 反传；默认 `fixed` 保持旧行为。Claude Code review 指出 prediction export 在 permutation 模式下必须记录 matching，且 padded slot 不能写有效误差；已修复并重跑 smoke / py_compile。

## S129 fixed vs permutation

S129 显示 `permutation_min` 没有改善，且显著低于 fixed reference：
- fixed val / test mask IoU = `3.699078e-01` / `4.244624e-01`
- permutation val / test mask IoU = `1.787238e-01` / `2.462286e-01`
- permutation val / test rotation MAE = `9.429713e+00` / `1.084679e+01` degree，劣于 fixed。

## 当前最佳配置

当前最佳 parametric 配置仍是 S115 raw MLP / shared head / fixed-order baseline。它没有达到 S117 oracle upper bound，但仍明显优于 dense V2 route 中反复出现的全背景 / 全前景塌缩。

## 下一步建议

- 不把 `permutation_min` 作为默认训练模式。
- 如果继续 parametric route，优先转向 forward consistency / differentiable rasterization 或 geometry-aware rotation/type objective，而不是继续 slot permutation loss。
- 如果继续做 set-style prediction，应先设计更明确的 slot decoder / query decoder，而不是仅用 loss-side permutation matching。
- 保留 S126/S127 导出的 per-sample diagnostics，用于后续定位 type-specific 或 rotation-specific failure。

## 自评

- S126-S130 已完成。
- 结论没有把 parametric route 说成最终成功；它仍是当前比 dense route 更有希望的诊断路线。
- 下一步不建议回到 dense mask margin / area / focal / sampling 盲扫。
