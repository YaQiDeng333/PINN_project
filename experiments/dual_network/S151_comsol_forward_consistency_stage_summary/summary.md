# S151 COMSOL forward consistency stage summary

## 目的

S151 收束 S145-S150，明确 learned forward surrogate 本身通过了 signal-level gate，但 simple forward consistency loss 没有改善 parametric mask IoU，因此暂不作为默认 objective。

## S147 结论

S147 训练了 geometry parameters -> multi-height Bz signal 的 learned forward surrogate。质量指标达到当前 consistency referee 的最低门槛：

| split | signal_nrmse_raw | signal_corr |
| --- | ---: | ---: |
| train | 3.767854e-01 | 9.258671e-01 |
| val | 5.026852e-01 | 8.657639e-01 |
| test | 4.577952e-01 | 8.886174e-01 |

因此 forward surrogate 可以继续作为 residual diagnostic / referee，但它仍只是 learned approximation，不等同 COMSOL solver。

## S149 结论

S149 比较了 parameter-only 与 learned forward consistency：

- `param_only_reference` train / val / test mask IoU = `6.980716e-01` / `3.699078e-01` / `4.244624e-01`。
- `forward_consistency_lambda01` train / val / test mask IoU = `5.947000e-01` / `3.103259e-01` / `3.954980e-01`。
- `forward_consistency_lambda1` train / val / test mask IoU = `4.301722e-01` / `2.477730e-01` / `3.392353e-01`。

`lambda=0.1` 对 rotation MAE 有局部改善，但没有改善 mask IoU；`lambda=1.0` 更明显退化。当前不应继续盲扫 forward consistency lambda。

## 当前判断

- Forward surrogate 仍有价值。
- Forward consistency loss 暂不作为默认 training objective。
- 下一步先诊断 residual 对 type / rotation / axis 错误是否敏感。
- 同时需要直接测试 type / rotation targeted supervision，而不是继续间接依赖 forward residual。

## 下一步

- S152: forward residual sensitivity diagnostic。
- S153: 在 parametric inverse runner 中增加 type / rotation targeted loss。
- S154: 运行 type / rotation targeted quick probe。

## 自评

- 没有把 forward consistency route 说成失败，只限定 simple loss 当前未改善。
- 明确保留 forward surrogate 作为 diagnostic referee。
- 下一步从机制诊断和更直接监督两个方向推进。
