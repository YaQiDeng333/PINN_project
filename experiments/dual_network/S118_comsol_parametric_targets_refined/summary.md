# S118 COMSOL parametric targets refined summary

## 目的

S118 对 S113 parametric targets 做训练目标改进：

- 将 `rotation_angle` 改为 `rotation_sin` / `rotation_cos`。
- 使用 train split 的 present components 统计量做 continuous normalization。
- 保留 raw continuous targets，用于真实单位 MAE 和 rasterization。
- 在训练 runner 中支持 inverse frequency type class weighting。

## refined schema

训练用 `target_schema`：

- `center_x`
- `center_y`
- `axis_x`
- `axis_y`
- `depth_or_shape_param`
- `rotation_sin`
- `rotation_cos`

保留的 `raw_target_schema`：

- `center_x`
- `center_y`
- `axis_x`
- `axis_y`
- `depth_or_shape_param`
- `rotation_angle`

## normalization

- train split 生成 `continuous_normalization_stats.npz`。
- val / test 使用 train stats，不使用自身统计量。
- `continuous_targets` 存储 normalized training target。
- `continuous_targets_raw` 存储真实单位 target。
- `continuous_targets_unscaled` 存储 sin/cos encoding 后、normalization 前的 target。

Train stats:

- mean = `[6.208817e-12, -2.546666e-04, 4.933339e-03, 5.864004e-03, 1.829166e-03, -5.111538e-02, 9.783491e-01]`
- std = `[1.203880e-02, 4.477250e-03, 4.988900e-04, 6.426300e-04, 4.244900e-04, 1.968572e-01, 3.828986e-02]`

## type vocab

`type_vocab = rectangular_notch, rotated_rect`

当前 V2 train / val / test 都包含这两个 type，训练 runner 可使用 `--type-class-weighting inverse_freq`。

## 与 S113 的差别

S113 使用 raw `rotation_angle` 并在 runner 内部做 continuous normalization。S118 将 angle transform 和 normalization stats 固化到 target artifact 中，减少训练 runner 对 target 语义的隐式假设。
