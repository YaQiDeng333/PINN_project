# S129 COMSOL parametric set-matching probe

## 目的

S129 比较 fixed-order component regression 与 `permutation_min` matching loss，判断 component slot/order ambiguity 是否限制当前 parametric route。

## 配置

共同配置：
- V2 S84 converted train/val/test NPZ
- S113 raw parametric targets
- `steps=3000`
- `encoder_type=mlp`
- `head_mode=shared`
- `hidden_dim=128`
- `latent_dim=64`
- `max_components=3`
- `lambda_presence=1.0`
- `lambda_type=1.0`
- `lambda_continuous=1.0`
- `export_predictions=true`

对比配置：
- `fixed_reference`: `component_matching_mode=fixed`
- `permutation_min`: `component_matching_mode=permutation_min`

## 结果

| config | split | presence_acc | type_acc | continuous_mae | rotation_mae | mask_iou |
|---|---:|---:|---:|---:|---:|---:|
| fixed_reference | train | `1.000000e+00` | `1.000000e+00` | `3.098074e-02` | `1.854687e-01` | `6.980716e-01` |
| fixed_reference | val | `1.000000e+00` | `6.500000e-01` | `1.289384e+00` | `7.731843e+00` | `3.699078e-01` |
| fixed_reference | test | `1.000000e+00` | `6.666667e-01` | `1.290734e+00` | `7.740397e+00` | `4.244624e-01` |
| permutation_min | train | `1.000000e+00` | `8.333333e-01` | `1.304149e+00` | `7.809930e+00` | `6.773069e-01` |
| permutation_min | val | `1.000000e+00` | `5.833333e-01` | `1.574556e+00` | `9.429713e+00` | `1.787238e-01` |
| permutation_min | test | `1.000000e+00` | `5.833333e-01` | `1.810269e+00` | `1.084679e+01` | `2.462286e-01` |

## 判断

`permutation_min` 没有改善 type、rotation 或 mask IoU，反而明显劣化 held-out performance。当前 fixed-order component sorting 不是最主要瓶颈；更可能的问题仍是 signal-to-geometry 泛化、type/rotation supervision 和 geometry-aware consistency。

## 自评

- 两组训练完成，metrics finite。
- 两组均未保存模型权重、checkpoint 或图片。
- `permutation_min` 运行成本显著高于 fixed，但没有带来收益。
