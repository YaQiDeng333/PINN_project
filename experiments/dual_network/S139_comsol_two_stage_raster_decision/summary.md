# S139 COMSOL two-stage raster decision

## S136 结论

S136 收束了 S131-S135：loss-side permutation matching 不作为主方向；differentiable rasterizer 和 raster BCE / Dice loss 已接入；S134 从训练初期加入 raster supervision 没有稳定超过 S115 raw MLP / shared head / fixed-order baseline。因此 raster loss 不应直接升为默认，而应先测试 parameter prefit 后的 fine-tune 口径。

## S137 新增能力

`train_comsol_parametric_inverse.py` 新增：

- `--raster-loss-start-step`：允许 raster BCE / Dice loss 在指定 step 后才启用。
- `--val-selection-metric none|val_mask_iou|val_loss`。
- `--val-selection-interval`。

默认配置仍保持旧行为：raster loss 默认关闭，validation selection 默认关闭。best endpoint 只用内存中的 `state_dict` deep copy 保存，不写出模型权重或 checkpoint。

Claude Code review 指出两个 must-fix：`val_loss` 与 delayed raster loss 会产生不可比较的 loss 组成；`val_mask_iou` selection 可能保留 pre-raster endpoint。已修复为：

- 禁止 `val_selection_metric=val_loss` 与 delayed raster loss 同时使用。
- delayed raster 阶段开始后重置 best tracking，并在 final step 也执行 selection 检查。

## S138 结果

| run | val mask_iou | test mask_iou | best_step | 当前判断 |
| --- | ---: | ---: | ---: | --- |
| `param_only_val_select` | 4.339882e-01 | 3.966467e-01 | 500 | validation selection 提升 val，但 test 低于 S115 |
| `two_stage_raster_dice` | 4.050472e-01 | 4.022032e-01 | 3000 | train geometry calibration 强，但 held-out 未超过 S115 |
| `two_stage_raster_bce_dice` | 2.810415e-01 | 3.105837e-01 | 2500 | 明显劣化 |

S115 / S134 parameter-only baseline 的 val / test mask IoU 为 `3.699078e-01` / `4.244624e-01`。S138 没有任何配置同时超过该 baseline；`param_only_val_select` 的 val 更高但 test 更低，说明 endpoint selection 对 val 有帮助但没有稳定泛化；`two_stage_raster_dice` 没有超过 S115 test。

## 当前最佳配置

当前最佳稳定 parametric 配置仍是 S115 / S134 `param_only_reference`：raw MLP / shared head / fixed-order / parameter-only objective，train / val / test mask IoU = `6.980716e-01` / `3.699078e-01` / `4.244624e-01`。

## 决策

Parametric route 继续，但 two-stage raster fine-tune 不能作为默认配置。当前证据更支持：

- 保留 `raster_loss_start_step` 和 `val_selection_metric` 作为诊断能力；
- 不继续盲扫 raster BCE / Dice 权重；
- 下一步优先转向 forward consistency / physics feature extraction；
- 如果继续 raster route，应测试 very short post-selection raster fine-tune，而不是长程 raster fine-tune。

## 自评

- S136-S139 已给出清晰阶段结论。
- 没有夸大 raster fine-tune：它改善 train calibration，但没有稳定改善 held-out baseline。
- 下一步建议明确指向 forward consistency / physics feature extraction，而不是 dense mask loss 或 raster 权重盲扫。
