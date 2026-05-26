# S136 COMSOL parametric raster-supervision stage summary

## S131 结论

S131 收束 S126-S130 后确认：

- prediction export 和 grouped diagnostics 已经可用于 sample/component 级错误定位；
- `permutation_min` 明显低于 fixed-order baseline；
- fixed-order raw MLP baseline 仍是当前最稳定配置；
- 下一步不继续 loss-side set matching，而是测试 differentiable raster mask supervision。

## S132/S133 结论

S132/S133 已完成：

- differentiable soft rasterizer；
- raster BCE / Dice loss 接入 `train_comsol_parametric_inverse.py`；
- `lambda_raster_bce` / `lambda_raster_dice` / `raster_softness_cells` / `raster_target_source`；
- smoke test 和 py_compile；
- Claude Code review。

Review 结论：

- S132 初始 rotation unit heuristic 有风险，已修复为显式 schema 口径；
- S133/S134 integration review 确认反归一化 geometry、target alignment 和默认兼容性正确，无 must-fix。

## S134 结论

S134 从训练初期直接加入 raster loss，没有稳定超过 parameter-only baseline：

- `param_only_reference` val / test mask IoU = `3.699078e-01` / `4.244624e-01`；
- `raster_dice1` val / test mask IoU = `3.523885e-01` / `4.385081e-01`；
- `raster_bce05_dice1` val / test mask IoU = `3.697576e-01` / `4.096553e-01`；
- `raster_dice1_soft2` val / test mask IoU = `3.577784e-01` / `4.088950e-01`。

`raster_dice1` 小幅提升 test，但 val 下降，不能设为默认。

## 当前判断

Raster loss 仍有价值，但更可能适合作为后期 fine-tune 约束，而不是从训练初期直接加入。下一步应测试 two-stage parameter prefit + raster fine-tune，并配合 validation-aware endpoint selection 避免最后阶段退化。

## 自评

- S131-S135 结论已准确收束。
- 没有把 raster supervision 说成已稳定成功。
- 下一步明确指向 S137-S139 two-stage raster fine-tune。
