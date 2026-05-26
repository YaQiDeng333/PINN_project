# S123 COMSOL parametric architecture quick gate

## 目的

用较小训练预算比较 shared / component-specific heads 与 MLP / CNN / attention signal encoder，避免直接进入 full V2 长实验。

数据使用 S84 V2 converted NPZ 与 S113 raw parametric targets。共同配置为 `steps=2000`、`hidden_dim=128`、`latent_dim=64`、`max_components=3`、`type_class_weighting=inverse_freq`、`lambda_type=1.5`、`lambda_rotation=3.0`。

## 结果

| config | split | presence_acc | type_acc | continuous_mae | center_mae | axis_mae | rotation_mae | depth_mae | mask_iou |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_mlp_shared_reference | train | 1.000000e+00 | 1.000000e+00 | 2.136699e-02 | 3.549952e-04 | 1.258343e-05 | 1.274630e-01 | 4.184182e-06 | 6.505209e-01 |
| raw_mlp_shared_reference | val | 1.000000e+00 | 6.333333e-01 | 1.071052e+00 | 1.706386e-03 | 6.217735e-04 | 6.421423e+00 | 2.329184e-04 | 3.122039e-01 |
| raw_mlp_shared_reference | test | 1.000000e+00 | 6.666667e-01 | 1.179666e+00 | 1.437900e-03 | 5.824498e-04 | 7.073723e+00 | 2.318654e-04 | 3.738216e-01 |
| raw_mlp_component_specific | train | 1.000000e+00 | 1.000000e+00 | 2.944517e-02 | 4.433426e-04 | 2.482071e-05 | 1.757249e-01 | 9.673855e-06 | 6.351883e-01 |
| raw_mlp_component_specific | val | 1.000000e+00 | 6.166667e-01 | 1.303563e+00 | 1.518775e-03 | 5.476592e-04 | 7.816954e+00 | 2.880564e-04 | 3.572195e-01 |
| raw_mlp_component_specific | test | 1.000000e+00 | 6.666667e-01 | 1.144282e+00 | 1.665895e-03 | 5.499509e-04 | 6.861016e+00 | 2.412747e-04 | 3.376996e-01 |
| raw_cnn_component_specific | train | 1.000000e+00 | 8.166667e-01 | 1.034491e+00 | 1.645317e-03 | 3.965958e-04 | 6.202634e+00 | 2.304497e-04 | 3.880497e-01 |
| raw_cnn_component_specific | val | 1.000000e+00 | 7.000000e-01 | 1.083696e+00 | 1.618631e-03 | 4.212738e-04 | 6.497802e+00 | 2.926545e-04 | 3.662237e-01 |
| raw_cnn_component_specific | test | 1.000000e+00 | 6.666667e-01 | 1.208580e+00 | 1.421880e-03 | 4.342486e-04 | 7.247536e+00 | 2.308462e-04 | 3.892232e-01 |
| raw_attention_component_specific | train | 1.000000e+00 | 9.633333e-01 | 7.680842e-01 | 1.527448e-03 | 3.416336e-04 | 4.604577e+00 | 1.901062e-04 | 4.040538e-01 |
| raw_attention_component_specific | val | 1.000000e+00 | 5.000000e-01 | 1.434788e+00 | 2.677976e-03 | 4.879210e-04 | 8.602077e+00 | 3.213805e-04 | 2.313885e-01 |
| raw_attention_component_specific | test | 1.000000e+00 | 5.000000e-01 | 1.441374e+00 | 1.997957e-03 | 4.935169e-04 | 8.643003e+00 | 2.600122e-04 | 2.771757e-01 |

## 与 S115 / S119 对比

- S115 raw parametric baseline: val / test mask IoU = `3.699078e-01` / `4.244624e-01`。
- S119 refined MLP: val / test mask IoU = `3.257646e-01` / `3.885092e-01`。
- S123 最佳 val/test mask IoU 配置是 `raw_cnn_component_specific`，val / test = `3.662237e-01` / `3.892232e-01`，未超过 S115。
- `raw_cnn_component_specific` 的 val type accuracy 达到 `7.000000e-01`，高于 S115 的 `6.500000e-01`，rotation MAE 也低于 S115，因此进入 S124 full probe。

## 当前判断

- component-specific heads 没有稳定提升 mask IoU。
- CNN1D encoder 对 val type / rotation 有一定帮助，但仍没有改善最终 held-out mask IoU。
- attention pooling 当前明显退化。
- S124 只对 `raw_cnn_component_specific` 做 longer run，不继续扩大其他配置。
