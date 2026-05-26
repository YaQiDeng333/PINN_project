# S163 COMSOL parametric center-aware loss quick gate

## 配置

三组均使用 S84 V2 converted NPZ + S113 raw targets，`steps=1500`、`lr=1e-3`、`hidden_dim=128`、`latent_dim=64`、`max_components=3`、`encoder_type=mlp`、`head_mode=shared`、`component_matching_mode=fixed`、`export_predictions=true`。Gate 只比较同一轮 `param_only_1500_reference`，S115 3000-step 仅作历史参考。

## Metrics

| config | split | presence_acc | type_acc | continuous_mae | center_mae | center_grid_mae | center_axis_relative_mae | axis_mae | rotation_mae | depth_mae | mask_iou |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| param_only_1500_reference | train | 1.000000 | 1.000000 | 0.076935 | 0.000377 | 1.832911 | 0.122668 | 0.000020 | 0.460801 | 0.000014 | 0.656469 |
| param_only_1500_reference | val | 1.000000 | 0.650000 | 1.235658 | 0.001307 | 7.535057 | 0.388393 | 0.000605 | 7.409863 | 0.000257 | 0.415913 |
| param_only_1500_reference | test | 1.000000 | 0.650000 | 1.250103 | 0.001274 | 7.113038 | 0.380114 | 0.000577 | 7.496665 | 0.000250 | 0.427099 |
| center_grid_loss | train | 1.000000 | 1.000000 | 0.332303 | 0.000025 | 0.129123 | 0.008196 | 0.000141 | 1.993406 | 0.000079 | 0.729657 |
| center_grid_loss | val | 1.000000 | 0.666667 | 1.170729 | 0.001076 | 6.082055 | 0.325022 | 0.000558 | 7.020877 | 0.000230 | 0.470171 |
| center_grid_loss | test | 1.000000 | 0.600000 | 1.417178 | 0.001016 | 5.564289 | 0.312888 | 0.000580 | 8.499659 | 0.000218 | 0.504389 |
| center_axis_relative | train | 1.000000 | 1.000000 | 0.050833 | 0.000072 | 0.395046 | 0.021929 | 0.000015 | 0.304815 | 0.000008 | 0.725207 |
| center_axis_relative | val | 1.000000 | 0.650000 | 1.226082 | 0.001520 | 9.384233 | 0.446056 | 0.000535 | 7.352133 | 0.000250 | 0.374787 |
| center_axis_relative | test | 1.000000 | 0.633333 | 1.687742 | 0.001607 | 10.066482 | 0.472576 | 0.000672 | 10.121602 | 0.000291 | 0.392738 |

## Gate decision

`center_grid_loss` passes S163 gate against the same 1500-step reference:

- Val mask IoU improves from `0.415913` to `0.470171` (`+0.054258`).
- Test mask IoU improves from `0.427099` to `0.504389` (`+0.077290`).
- Val/test center grid and axis-relative errors both decrease.
- Presence remains `1.0`; type/axis/rotation tradeoffs exist but do not prevent mask-IoU improvement.

`center_axis_relative` fails the gate because val/test mask IoU and center errors are worse than the same-run reference.

## 自评

S163 shows center-aware grid loss can convert lower center error into higher held-out mask IoU in the quick gate. Because val and test both improved, S164 full probe should run for `center_grid_loss`.
