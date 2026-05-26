# S149 COMSOL parametric forward-consistency inverse probe

## 目的

S149 比较 parameter-only baseline 与 learned forward consistency，判断 geometry -> Bz surrogate residual 是否能改善 held-out type / rotation / geometry 泛化和最终 mask IoU。

## 配置

共同数据：

- S84 V2 converted NPZ
- S113 raw parametric targets
- inverse model: raw MLP / shared head / fixed-order
- `hidden_dim=128`
- `latent_dim=64`
- `max_components=3`

三组：

1. `param_only_reference`: 原 parameter-only baseline，`steps=3000`
2. `forward_consistency_lambda01`: forward pretrain `3000` steps，inverse `3000` steps，`lambda_forward_consistency=0.1`
3. `forward_consistency_lambda1`: forward pretrain `3000` steps，inverse `3000` steps，`lambda_forward_consistency=1.0`

## 结果

| config | split | presence_acc | type_acc | continuous_mae | center_mae | axis_mae | rotation_mae | depth_mae | mask_iou | forward_nrmse | forward_corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| param_only_reference | train | 1.000000e+00 | 1.000000e+00 | 3.098074e-02 | 1.978452e-04 | 7.397077e-06 | 1.854687e-01 | 5.010099e-06 | 6.980716e-01 | n/a | n/a |
| param_only_reference | val | 1.000000e+00 | 6.500000e-01 | 1.289384e+00 | 1.485341e-03 | 5.920604e-04 | 7.731843e+00 | 3.065882e-04 | 3.699078e-01 | n/a | n/a |
| param_only_reference | test | 1.000000e+00 | 6.666667e-01 | 1.290734e+00 | 1.285206e-03 | 5.781414e-04 | 7.740397e+00 | 2.824507e-04 | 4.244624e-01 | n/a | n/a |
| forward_consistency_lambda01 | train | 1.000000e+00 | 1.000000e+00 | 7.265875e-01 | 5.083076e-04 | 1.090395e-05 | 4.358479e+00 | 6.868952e-06 | 5.947000e-01 | 3.687188e-01 | 9.289096e-01 |
| forward_consistency_lambda01 | val | 1.000000e+00 | 6.333333e-01 | 1.096440e+00 | 1.808025e-03 | 5.540216e-04 | 6.573701e+00 | 2.152385e-04 | 3.103259e-01 | 6.664540e-01 | 7.720926e-01 |
| forward_consistency_lambda01 | test | 1.000000e+00 | 6.500000e-01 | 1.185092e+00 | 1.327344e-03 | 5.595572e-04 | 7.106478e+00 | 2.996631e-04 | 3.954980e-01 | 5.632792e-01 | 8.318901e-01 |
| forward_consistency_lambda1 | train | 1.000000e+00 | 1.000000e+00 | 1.179633e+00 | 1.135215e-03 | 3.581133e-05 | 7.075432e+00 | 2.423720e-05 | 4.301722e-01 | 3.657509e-01 | 9.300855e-01 |
| forward_consistency_lambda1 | val | 1.000000e+00 | 6.166667e-01 | 1.193903e+00 | 2.319555e-03 | 5.746344e-04 | 7.157398e+00 | 2.325334e-04 | 2.477730e-01 | 5.195281e-01 | 8.566328e-01 |
| forward_consistency_lambda1 | test | 1.000000e+00 | 5.833333e-01 | 1.199674e+00 | 1.788043e-03 | 6.068033e-04 | 7.192984e+00 | 2.692388e-04 | 3.392353e-01 | 5.306449e-01 | 8.537060e-01 |

## 判断

- `lambda=0.1` 对 val/test rotation MAE 有局部改善，但 mask IoU 明显低于 parameter-only baseline。
- `lambda=1.0` forward constraint 过强，train/val/test mask IoU 均明显退化，并伤害 test type accuracy。
- Forward consistency 在当前实现中没有超过 S115 / S143 raw baseline。
- 当前最佳 parametric 配置仍是 raw MLP / shared head / fixed-order parameter-only baseline。

## 自评

- S149 完整跑完三组，metrics finite。
- 结果不支持把 learned forward consistency 设为默认 objective。
- 需要区分两个原因：surrogate residual 可能更偏 signal reconstruction，而不直接对应 mask IoU；也可能当前 surrogate / inverse 联训方式对 geometry 参数造成过强约束。
