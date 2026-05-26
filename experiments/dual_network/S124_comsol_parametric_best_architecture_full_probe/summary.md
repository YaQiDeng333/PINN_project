# S124 COMSOL parametric best architecture full probe

## 执行原因

S123 中 `raw_cnn_component_specific` 是 quick gate 的最佳 val/test mask IoU 配置，并且 val type accuracy 与 rotation MAE 相比 S115 有改善，因此执行 full probe。

## 配置

- encoder_type: `cnn1d`
- head_mode: `component_specific`
- targets: S113 raw parametric targets
- steps: `6000`
- lr: `1e-3`
- hidden_dim: `128`
- latent_dim: `64`
- max_components: `3`
- type_class_weighting: `inverse_freq`
- lambda_type: `1.5`
- lambda_rotation: `3.0`

## 结果

| split | presence_acc | type_acc | continuous_mae | center_mae | axis_mae | rotation_mae | depth_mae | mask_iou |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1.000000e+00 | 9.766667e-01 | 8.985017e-01 | 1.528779e-03 | 3.428130e-04 | 5.387074e+00 | 1.923422e-04 | 3.831555e-01 |
| val | 1.000000e+00 | 6.333333e-01 | 1.188368e+00 | 1.852659e-03 | 4.471446e-04 | 7.125315e+00 | 2.928631e-04 | 3.133465e-01 |
| test | 1.000000e+00 | 6.333333e-01 | 1.420662e+00 | 1.909040e-03 | 4.800996e-04 | 8.518946e+00 | 2.482069e-04 | 3.153288e-01 |

## 与 S115 / S123 对比

- S115 raw baseline val / test mask IoU = `3.699078e-01` / `4.244624e-01`。
- S123 `raw_cnn_component_specific` val / test mask IoU = `3.662237e-01` / `3.892232e-01`。
- S124 longer run val / test mask IoU 降到 `3.133465e-01` / `3.153288e-01`。

## 当前判断

`raw_cnn_component_specific_longer` 不应作为 parametric route 默认配置。Longer CNN/component-specific training 没有改善 held-out mask IoU，反而降低 test type accuracy 和 mask IoU。
