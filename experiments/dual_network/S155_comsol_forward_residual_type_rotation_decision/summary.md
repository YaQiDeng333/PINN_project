# S155 COMSOL forward residual and type/rotation decision

## S151 forward consistency 结论

S151 确认 S147 forward surrogate 质量可用，但 S149 simple forward consistency loss 没有改善 mask IoU，因此不作为默认 objective。

## S152 residual sensitivity 结论

Forward residual 对 rotation perturbation 非常敏感，对 type swap 中等敏感，但对 axis scaling 基本不敏感。Predicted geometry residual 与 mask IoU 的相关性有限：

- val: `-2.503890e-01`
- test: `-4.378152e-01`

因此 learned forward residual 更适合作为 diagnostic referee，而不是当前默认训练 loss。

## S153/S154 targeted supervision 结果

S153 新增 `lambda_type_extra`、`lambda_rotation_extra` 和 `rotation_loss_mode`。S154 quick probe 显示：

- `type_extra` 没有改善 type accuracy，val mask IoU 明显下降；test mask IoU 只极小超过 S115，不构成稳定改善。
- `rotation_extra` 改善 train rotation fit 和 val mask IoU，但 test mask IoU 低于 S115。
- `type_rotation_extra` 改善 val rotation / mask，但 test 明显下降。

## 当前最佳配置

当前最佳稳定配置仍是 S115 / S143 / S149 / S154 `param_only_reference`：

- raw MLP
- shared head
- fixed-order component targets
- parameter-only loss

S154 中 `type_extra` 的 test mask IoU 略高于 S115，但 val 和 type accuracy 退化，因此不能作为新默认。

## 是否继续 forward consistency

不建议继续盲扫 forward consistency loss。Forward surrogate 可继续用于 residual diagnostic 或 offline consistency ranking，但当前不作为默认 training objective。

## 是否继续 type/rotation targeted loss

不建议按当前简单 extra-loss 形式继续盲扫。Rotation targeted loss 有局部信号，可以作为后续更精细设计的参考；type extra CE 当前没有改善 held-out type。

## 下一步建议

- 如果继续 parametric route，应考虑更强 target representation，例如 Piao-style profile parameterization 或 component-specific shape/profile descriptors。
- 需要检查数据层面的 type / rotation balance，必要时生成更多 type/rotation-balanced COMSOL data。
- 如果继续使用 forward surrogate，应改为 offline ranking / residual diagnostic，或设计 dimension-specific residual，而不是简单加到 total loss。
- 如果继续监督 type/rotation，建议做 class-balanced sampling、type-specific heads 或 auxiliary component classifier，而不是单纯加大 CE 权重。

## 自评

- S151-S155 没有回到 dense mask runner。
- 没有保存模型权重、checkpoint 或图片。
- 当前结论明确：S115 raw parameter-only baseline 仍是稳定 best；forward residual 和 targeted losses 都提供机制信息，但未形成新默认。
