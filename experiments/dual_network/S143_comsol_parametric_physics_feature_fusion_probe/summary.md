# S143 COMSOL parametric physics feature fusion probe

## 目的

S143 比较 raw signal、physics features、raw+features 三种输入，判断 S141 的 MFL physics features 是否改善 parametric inverse 的 held-out type、rotation 和 mask IoU。

共同配置：S84 COMSOL V2 converted NPZ + S113 raw parametric targets，`steps=3000`、`lr=1e-3`、`hidden_dim=128`、`latent_dim=64`、`max_components=3`、`encoder_type=mlp`、`head_mode=shared`、`component_matching_mode=fixed`、`export_predictions=true`。

## 配置

| run | feature_fusion_mode | feature_dim |
| --- | --- | ---: |
| `raw_signal_reference` | `none` | 0 |
| `physics_features_only` | `features_only` | 58 |
| `raw_plus_physics_features` | `concat_latent` | 58 |

## 指标

| run | split | presence_acc | type_acc | continuous_mae | center_mae | axis_mae | rotation_mae | depth_mae | mask_iou |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_signal_reference` | train | 1.000000e+00 | 1.000000e+00 | 3.098074e-02 | 1.978452e-04 | 7.397077e-06 | 1.854687e-01 | 5.010099e-06 | 6.980716e-01 |
| `raw_signal_reference` | val | 1.000000e+00 | 6.500000e-01 | 1.289384e+00 | 1.485341e-03 | 5.920604e-04 | 7.731843e+00 | 3.065882e-04 | 3.699078e-01 |
| `raw_signal_reference` | test | 1.000000e+00 | 6.666667e-01 | 1.290734e+00 | 1.285206e-03 | 5.781414e-04 | 7.740397e+00 | 2.824507e-04 | 4.244624e-01 |
| `physics_features_only` | train | 1.000000e+00 | 1.000000e+00 | 4.692168e-03 | 3.382475e-05 | 2.103292e-06 | 2.807922e-02 | 1.779789e-06 | 7.273660e-01 |
| `physics_features_only` | val | 1.000000e+00 | 6.500000e-01 | 1.498954e+00 | 2.533332e-03 | 5.409532e-04 | 8.987270e+00 | 3.072060e-04 | 2.362846e-01 |
| `physics_features_only` | test | 1.000000e+00 | 5.500000e-01 | 1.831322e+00 | 2.435295e-03 | 5.577851e-04 | 1.098153e+01 | 4.154678e-04 | 2.327774e-01 |
| `raw_plus_physics_features` | train | 1.000000e+00 | 1.000000e+00 | 5.225517e-02 | 2.866366e-04 | 1.114007e-05 | 3.129291e-01 | 6.301345e-06 | 6.756908e-01 |
| `raw_plus_physics_features` | val | 1.000000e+00 | 6.666667e-01 | 1.024911e+00 | 1.707701e-03 | 5.610666e-04 | 6.144658e+00 | 2.723982e-04 | 3.313752e-01 |
| `raw_plus_physics_features` | test | 1.000000e+00 | 5.833333e-01 | 1.480355e+00 | 1.896865e-03 | 5.434547e-04 | 8.876973e+00 | 2.775385e-04 | 3.051455e-01 |

## 对比

- `physics_features_only` 的 train mask IoU 达到 `7.273660e-01`，接近 / 略高于 S117 train oracle 平均，但 val/test 明显退化，说明 handcrafted features 可以强拟合 train，但泛化不足。
- `raw_plus_physics_features` 改善 val type accuracy 和 val rotation MAE，但没有改善 val/test mask IoU，test type 和 rotation 也低于 raw reference。
- `raw_signal_reference` 完全复现 S115 baseline，仍是当前最佳稳定配置。

## 判断

S141 physics features 有 train-side 信号，但当前 feature-only 或 concat fusion 没有稳定改善 held-out type / rotation / mask IoU。当前瓶颈不只是 raw MLP 缺少显式 peak/decay features；更可能需要 forward consistency、physics-aware supervision、feature regularization 或更直接的 signal-to-geometry inductive bias。

## 自评

- 三组均完成，metrics finite。
- 没有发现 feature normalization 或 shape 对齐异常。
- 当前不建议把 physics feature fusion 作为默认配置。
