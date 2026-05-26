# S154 COMSOL type / rotation targeted supervision probe

## 目的

S154 测试更直接的 type / rotation targeted supervision 是否改善 held-out type / rotation / mask IoU。

## 配置

共同配置：

- S84 V2 converted NPZ + S113 raw targets
- raw MLP / shared head / fixed-order
- `steps=3000`
- `hidden_dim=128`
- `latent_dim=64`
- `lambda_presence=1.0`
- `lambda_type=1.0`
- `lambda_continuous=1.0`
- `export_predictions=true`

四组：

1. `param_only_reference`: no extra type / rotation loss
2. `type_extra`: `lambda_type_extra=1.0`
3. `rotation_extra`: `lambda_rotation_extra=2.0`, `rotation_loss_mode=circular`
4. `type_rotation_extra`: both extra losses enabled

## 结果

| config | split | presence_acc | type_acc | continuous_mae | center_mae | axis_mae | rotation_mae | depth_mae | mask_iou |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| param_only_reference | train | 1.000000e+00 | 1.000000e+00 | 3.098074e-02 | 1.978452e-04 | 7.397077e-06 | 1.854687e-01 | 5.010099e-06 | 6.980716e-01 |
| param_only_reference | val | 1.000000e+00 | 6.500000e-01 | 1.289384e+00 | 1.485341e-03 | 5.920604e-04 | 7.731843e+00 | 3.065882e-04 | 3.699078e-01 |
| param_only_reference | test | 1.000000e+00 | 6.666667e-01 | 1.290734e+00 | 1.285206e-03 | 5.781414e-04 | 7.740397e+00 | 2.824507e-04 | 4.244624e-01 |
| type_extra | train | 1.000000e+00 | 1.000000e+00 | 3.955308e-02 | 2.670472e-04 | 7.760472e-06 | 2.367634e-01 | 5.381066e-06 | 6.770008e-01 |
| type_extra | val | 1.000000e+00 | 6.333333e-01 | 1.190836e+00 | 1.862306e-03 | 5.903560e-04 | 7.139857e+00 | 2.511544e-04 | 2.987030e-01 |
| type_extra | test | 1.000000e+00 | 6.333333e-01 | 1.336302e+00 | 1.166588e-03 | 6.027818e-04 | 8.013975e+00 | 3.005727e-04 | 4.268750e-01 |
| rotation_extra | train | 1.000000e+00 | 1.000000e+00 | 1.568230e-03 | 1.664837e-04 | 4.559708e-06 | 9.065552e-03 | 2.216611e-06 | 7.059365e-01 |
| rotation_extra | val | 1.000000e+00 | 6.166667e-01 | 1.294458e+00 | 1.338018e-03 | 5.259672e-04 | 7.762744e+00 | 2.776451e-04 | 3.953165e-01 |
| rotation_extra | test | 1.000000e+00 | 6.166667e-01 | 1.250435e+00 | 1.306602e-03 | 5.243988e-04 | 7.498660e+00 | 2.879902e-04 | 4.134323e-01 |
| type_rotation_extra | train | 1.000000e+00 | 1.000000e+00 | 3.593546e-02 | 6.556689e-04 | 8.455513e-05 | 2.140695e-01 | 6.290638e-05 | 5.703712e-01 |
| type_rotation_extra | val | 1.000000e+00 | 6.166667e-01 | 1.160516e+00 | 1.416306e-03 | 5.819604e-04 | 6.958868e+00 | 2.328613e-04 | 3.817137e-01 |
| type_rotation_extra | test | 1.000000e+00 | 6.000000e-01 | 1.325704e+00 | 1.627835e-03 | 5.297940e-04 | 7.949647e+00 | 2.626113e-04 | 3.627724e-01 |

## 判断

- `type_extra` 没有改善 type accuracy；val mask IoU 明显下降，test mask IoU 只比 baseline 高约 `2.4e-03`，不构成稳定改善。
- `rotation_extra` 显著改善 train rotation fit，并提高 val mask IoU 到 `3.953165e-01`，但 test mask IoU 低于 S115 baseline。
- `type_rotation_extra` 改善 val rotation MAE 和 val mask IoU，但 test 明显下降。
- 当前没有一组稳定超过 S115 / S143 parameter-only baseline。

## 自评

- 四组均完成，metrics finite。
- Targeted type supervision 没有直接解决 held-out type 泛化。
- Rotation targeted loss 有一定 val-side signal，但没有稳定转化为 test mask IoU。
