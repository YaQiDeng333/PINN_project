# S113 COMSOL parametric targets aggregate summary

## 数据来源

S113 从 S84 V2 raw `defect_params.csv` 构造 component-level parametric targets，并与 S84 converted NPZ 的 split-local sample order 对齐。

## 输出

- train: `experiments/dual_network/S113_comsol_parametric_targets/train/parametric_targets.npz`
- val: `experiments/dual_network/S113_comsol_parametric_targets/val/parametric_targets.npz`
- test: `experiments/dual_network/S113_comsol_parametric_targets/test/parametric_targets.npz`

每个 split 还包含：
- `parametric_target_summary.md`
- `parametric_target_preview.csv`

## target schema

`continuous_targets` schema:

- `center_x`
- `center_y`
- `axis_x`
- `axis_y`
- `depth_or_shape_param`
- `rotation_angle`

其他字段：

- `presence_targets`: `[N,3]`
- `type_targets`: `[N,3]`
- `sample_indices`: `[N]`
- `type_vocab`: `rectangular_notch`, `rotated_rect`

## split 汇总

| split | samples | max_components | component count | type_vocab |
| --- | ---: | ---: | --- | --- |
| train | 100 | 3 | all samples have 3 components | rectangular_notch, rotated_rect |
| val | 20 | 3 | all samples have 3 components | rectangular_notch, rotated_rect |
| test | 20 | 3 | all samples have 3 components | rectangular_notch, rotated_rect |

## 参数范围

三个 split 的参数范围一致或高度重叠：

- `center_x`: -0.0202 到 0.0202
- `center_y`: -0.0057 到 0.005
- `axis_x`: 0.004 到 0.0056
- `axis_y`: 0.005 到 0.0066
- `depth_or_shape_param`: 0.001 到 0.0025
- `rotation_angle`: -30 到 30 degree

## 当前判断

- `max_components=3` 适合当前 V2 fallback 数据；没有样本被截断。
- `type_vocab` 在 train / val / test 中一致，可用于 S115 parametric inverse training。
- 第一版 target 使用 `source_component_json` 中的 component-level fields，避免将 multi_defect 压成 sample-level aggregate target。
