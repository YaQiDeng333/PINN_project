# S138 COMSOL parametric two-stage raster fine-tune probe

## 目的

本阶段比较 parameter-only baseline 与 two-stage raster fine-tune，判断 differentiable raster loss 是否更适合作为后期 fine-tune 约束，而不是从训练初期直接加入。

数据使用 S84 COMSOL V2 converted NPZ 与 S113 raw parametric targets。共同配置为 `encoder_type=mlp`、`head_mode=shared`、`component_matching_mode=fixed`、`steps=3000`、`lr=1e-3`、`hidden_dim=128`、`latent_dim=64`、`max_components=3`、`export_predictions=true`，并启用 `val_selection_metric=val_mask_iou`、`val_selection_interval=500`。

## 配置

| run | raster_bce | raster_dice | raster_loss_start_step | best_step | best_val_mask_iou |
| --- | ---: | ---: | ---: | ---: | ---: |
| `param_only_val_select` | 0.0 | 0.0 | 0 | 500 | 4.339882e-01 |
| `two_stage_raster_dice` | 0.0 | 1.0 | 2000 | 3000 | 4.050472e-01 |
| `two_stage_raster_bce_dice` | 0.25 | 1.0 | 2000 | 2500 | 2.810415e-01 |

## 指标

| run | split | presence_acc | type_acc | continuous_mae | center_mae | axis_mae | rotation_mae | depth_mae | mask_iou |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `param_only_val_select` | train | 1.000000e+00 | 1.000000e+00 | 1.913730e-01 | 6.412975e-04 | 9.515968e-05 | 1.146696e+00 | 6.864709e-05 | 5.745143e-01 |
| `param_only_val_select` | val | 1.000000e+00 | 6.500000e-01 | 1.239756e+00 | 1.218251e-03 | 6.014539e-04 | 7.434644e+00 | 2.505824e-04 | 4.339882e-01 |
| `param_only_val_select` | test | 1.000000e+00 | 6.666667e-01 | 1.294316e+00 | 1.354578e-03 | 5.606438e-04 | 7.761821e+00 | 2.466869e-04 | 3.966467e-01 |
| `two_stage_raster_dice` | train | 1.000000e+00 | 1.000000e+00 | 1.081236e-01 | 9.027575e-05 | 8.051978e-05 | 6.483752e-01 | 2.521608e-05 | 7.427683e-01 |
| `two_stage_raster_dice` | val | 1.000000e+00 | 6.500000e-01 | 1.382037e+00 | 1.281206e-03 | 5.994443e-04 | 8.288183e+00 | 2.759393e-04 | 4.050472e-01 |
| `two_stage_raster_dice` | test | 1.000000e+00 | 6.666667e-01 | 1.352693e+00 | 1.430968e-03 | 5.894130e-04 | 8.111882e+00 | 2.380988e-04 | 4.022032e-01 |
| `two_stage_raster_bce_dice` | train | 1.000000e+00 | 1.000000e+00 | 1.552390e-01 | 3.268475e-03 | 9.069645e-05 | 9.246728e-01 | 4.285114e-05 | 5.686725e-01 |
| `two_stage_raster_bce_dice` | val | 1.000000e+00 | 6.500000e-01 | 1.325058e+00 | 4.125326e-03 | 5.971059e-04 | 7.940632e+00 | 2.723842e-04 | 2.810415e-01 |
| `two_stage_raster_bce_dice` | test | 1.000000e+00 | 6.166667e-01 | 1.350409e+00 | 3.553443e-03 | 5.438879e-04 | 8.094011e+00 | 2.443827e-04 | 3.105837e-01 |

## 判断

- `param_only_val_select` 提高了 val mask IoU，但 test mask IoU 低于 S115 / S134 parameter-only baseline。
- `two_stage_raster_dice` 明显提高 train mask IoU，并略高于 `param_only_val_select` 的 test mask IoU，但 val/test 仍低于 S115 baseline。
- `two_stage_raster_bce_dice` 明显劣化 val/test，BCE 组合不适合作为当前 fine-tune 默认。
- two-stage raster fine-tune 没有稳定超过 S115 raw MLP baseline；raster loss 仍可能用于 train-side geometry calibration，但当前不能作为默认路线。

## 自评

- 三组均完成，metrics 为 finite。
- validation-aware selection 正常记录 `best_step` / `best_val_mask_iou`，且 best state 只保存在内存中。
- two-stage raster 没有达到替代 S115 baseline 的证据，下一步应转向 forward consistency / physics feature extraction，或 very short post-selection raster fine-tune。
