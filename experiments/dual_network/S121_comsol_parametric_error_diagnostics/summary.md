# S121 COMSOL parametric error diagnostics

## 目的

拆解 S115 raw parametric inverse 与 S119 refined MLP 的误差来源，判断下一步应优先修改 type / rotation / component head、signal encoder、loss，还是 rasterizer。

## 输入

- S115 run: `experiments/dual_network/S115_comsol_parametric_inverse_training_probe/v2_parametric_inverse`
- S119 run: `experiments/dual_network/S119_comsol_parametric_inverse_refined_probe/refined_mlp`
- Oracle rasterization: `experiments/dual_network/S117_comsol_parametric_raster_oracle`

当前 run 目录没有保存 per-sample predictions，因此本阶段只做 aggregate decomposition；未伪造 type / rotation / area bins。

## S115 raw vs oracle

| split | mask_iou | oracle_iou | oracle_gap | type_acc | rotation_mae |
|---|---:|---:|---:|---:|---:|
| train | 6.980716e-01 | 7.229967e-01 | 2.492512e-02 | 1.000000e+00 | 1.854690e-01 |
| val | 3.699078e-01 | 7.232882e-01 | 3.533803e-01 | 6.500000e-01 | 7.731843e+00 |
| test | 4.244624e-01 | 7.165838e-01 | 2.921214e-01 | 6.666667e-01 | 7.740396e+00 |

S115 train 已接近 oracle，但 val/test oracle gap 很大。主要 held-out 问题不是 presence，而是 type generalization、rotation generalization 和 geometry-to-mask 定位误差。

## S119 refined vs oracle

| split | mask_iou | oracle_iou | oracle_gap | type_acc | rotation_mae |
|---|---:|---:|---:|---:|---:|
| train | 6.689502e-01 | 7.229967e-01 | 5.404654e-02 | 1.000000e+00 | 2.569852e-01 |
| val | 3.257646e-01 | 7.232882e-01 | 3.975235e-01 | 6.166667e-01 | 7.278932e+00 |
| test | 3.885092e-01 | 7.165838e-01 | 3.280746e-01 | 6.833333e-01 | 7.859528e+00 |

S119 refined schema 没有缩小 held-out oracle gap。test type accuracy 小幅提升，但 val type accuracy 和 val/test mask IoU 均下降。

## 当前结论

- presence 已基本解决，不是当前主要瓶颈。
- train split 接近 oracle，说明 parametric runner 可拟合训练集，并且 rasterizer 上限足够。
- val/test 与 oracle 的 gap 明显，说明泛化瓶颈主要在 `Bz -> component geometry`，不是 target/mask schema。
- type accuracy 约 0.62 到 0.68，rotation MAE 约 7 到 8 degree，是下一步最明确的误差维度。
- S119 的 sin/cos + normalization 没有整体改善，单纯 target transform 不够。

## 下一步依据

S122 应优先测试：

- component-specific heads，降低 component slot 之间的输出层干扰；
- CNN1D / attention signal encoder，增强 multi-height Bz signal 的局部模式提取；
- type / rotation loss 分解和 rotation group 权重；
- 后续如果仍有 oracle gap，再考虑 forward consistency 或 differentiable rasterization。
