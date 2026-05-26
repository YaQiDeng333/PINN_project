# S140 COMSOL parametric raster fine-tune stage summary

## 目的

S140 收束 S136-S139，确认 differentiable raster loss 不再作为当前主攻方向的盲扫对象，并把下一步转向 physics-based MFL signal features 与 parametric inverse feature fusion。

## S136-S139 结论

- S136 确认 raster loss 从训练初期直接加入没有稳定超过 S115 / S134 parameter-only baseline。
- S137 增加 `raster_loss_start_step` 和 validation-aware endpoint selection，并修复 Claude Code review 指出的 delayed raster loss 与 `val_loss` selection 不可比问题。
- S138 的 `param_only_val_select` 只改善 val endpoint，test mask IoU 低于 S115。
- S138 的 `two_stage_raster_dice` 提高 train mask IoU，但 val / test 仍未超过 S115。
- S138 的 `two_stage_raster_bce_dice` 明显劣化。
- 当前最佳稳定配置仍是 S115 / S134 `param_only_reference` raw MLP / shared head / fixed-order baseline。

S115 baseline train / val / test mask IoU 为 `6.980716e-01` / `3.699078e-01` / `4.244624e-01`。

## 当前判断

Differentiable rasterizer 仍是有价值工具，可以用于 geometry / mask calibration 或后续 forward consistency 的近似组件。但 S136-S139 没有证明 raster BCE / Dice loss 权重或 two-stage fine-tune 能稳定改善 held-out mask IoU，因此当前不继续盲扫 raster loss。

当前更合理的瓶颈假设是：raw MLP signal encoder 没有稳定提取 multi-height MFL Bz 中的峰值、峰位、峰宽、能量和 lift-off 衰减等物理特征，导致 held-out type / rotation / geometry 泛化不足。

## 下一步

- S141：从 COMSOL V2 multi-height Bz signals 中提取 physics-based MFL features。
- S142：将 physics features 接入 `ParametricInverseNet`，支持 `features_only` 和 `concat_latent` fusion。
- S143：比较 raw signal、physics features、raw+features。

## 自评

- 总结没有把 raster fine-tune 说成失败，只限定为当前 quick gate 未稳定超过 baseline。
- 下一步明确转向 signal feature representation，而不是 dense mask runner、permutation matching 或 raster loss 权重盲扫。
