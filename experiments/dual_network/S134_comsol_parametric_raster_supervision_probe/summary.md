# S134 COMSOL parametric raster-supervision quick gate

## 目的

S134 比较参数-only baseline 与 differentiable raster mask supervision，判断 mask-level supervision 是否改善 V2 parametric route 的 held-out mask IoU。

## 共同配置

- V2 S84 converted train / val / test NPZ
- S113 raw parametric targets
- `steps=3000`
- `lr=1e-3`
- `hidden_dim=128`
- `latent_dim=64`
- `max_components=3`
- `encoder_type=mlp`
- `head_mode=shared`
- `component_matching_mode=fixed`
- `lambda_presence=1.0`
- `lambda_type=1.0`
- `lambda_continuous=1.0`
- `export_predictions=true`

## 结果

| config | split | presence_acc | type_acc | continuous_mae | center_mae | axis_mae | rotation_mae | depth_mae | mask_iou |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| param_only_reference | train | `1.000000e+00` | `1.000000e+00` | `3.098074e-02` | `1.978452e-04` | `7.397077e-06` | `1.854687e-01` | `5.010099e-06` | `6.980716e-01` |
| param_only_reference | val | `1.000000e+00` | `6.500000e-01` | `1.289384e+00` | `1.485341e-03` | `5.920604e-04` | `7.731843e+00` | `3.065882e-04` | `3.699078e-01` |
| param_only_reference | test | `1.000000e+00` | `6.666667e-01` | `1.290734e+00` | `1.285206e-03` | `5.781414e-04` | `7.740397e+00` | `2.824507e-04` | `4.244624e-01` |
| raster_dice1 | train | `1.000000e+00` | `1.000000e+00` | `4.483610e-02` | `1.591278e-03` | `6.767623e-05` | `2.656904e-01` | `7.928076e-06` | `6.876791e-01` |
| raster_dice1 | val | `1.000000e+00` | `6.333333e-01` | `1.144526e+00` | `3.173575e-03` | `5.865519e-04` | `6.859364e+00` | `2.739878e-04` | `3.523885e-01` |
| raster_dice1 | test | `1.000000e+00` | `6.833333e-01` | `1.179171e+00` | `2.634419e-03` | `5.837005e-04` | `7.068331e+00` | `2.579566e-04` | `4.385081e-01` |
| raster_bce05_dice1 | train | `1.000000e+00` | `1.000000e+00` | `4.704830e-02` | `2.322040e-03` | `7.010982e-05` | `2.775014e-01` | `4.329282e-06` | `5.899865e-01` |
| raster_bce05_dice1 | val | `1.000000e+00` | `6.166667e-01` | `1.261819e+00` | `2.709928e-03` | `6.224004e-04` | `7.563980e+00` | `2.690530e-04` | `3.697576e-01` |
| raster_bce05_dice1 | test | `1.000000e+00` | `6.833333e-01` | `1.179941e+00` | `2.622429e-03` | `5.792672e-04` | `7.072973e+00` | `2.714364e-04` | `4.096553e-01` |
| raster_dice1_soft2 | train | `1.000000e+00` | `1.000000e+00` | `4.430571e-02` | `1.103824e-03` | `4.150471e-05` | `2.635347e-01` | `8.762097e-06` | `6.964484e-01` |
| raster_dice1_soft2 | val | `1.000000e+00` | `6.333333e-01` | `1.260729e+00` | `2.711263e-03` | `5.461648e-04` | `7.557579e+00` | `2.807274e-04` | `3.577784e-01` |
| raster_dice1_soft2 | test | `1.000000e+00` | `6.666667e-01` | `1.351610e+00` | `2.629335e-03` | `5.687058e-04` | `8.102981e+00` | `2.806526e-04` | `4.088950e-01` |

## 判断

- `raster_dice1` 是 S134 内部最好的 raster-supervised 配置，test mask IoU 从 S115 baseline `4.244624e-01` 提升到 `4.385081e-01`。
- 但 `raster_dice1` 的 val mask IoU 从 `3.699078e-01` 降到 `3.523885e-01`，没有形成稳定 held-out 改善。
- `raster_bce05_dice1` 接近 val baseline，但 test 低于 baseline。
- `raster_dice1_soft2` 没有超过 baseline。

## 自评

- 四组训练均完成，metrics finite。
- 未保存模型权重、checkpoint 或图片。
- 中途发现 raster loss 应使用反归一化 geometry，已修复并重跑所有 raster-supervised runs；param-only reference 未受影响。
- Claude Code review 确认修复后接入无 must-fix。
