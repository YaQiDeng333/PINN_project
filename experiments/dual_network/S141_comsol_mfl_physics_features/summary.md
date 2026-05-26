# S141 COMSOL MFL physics features

## 目的

S141 从 COMSOL V2 multi-height Bz signals 中提取轻量 physics-based MFL features，用于后续 parametric inverse feature fusion。

## 输出

| split | features shape | feature count | finite |
| --- | ---: | ---: | --- |
| train | `[100, 58]` | 58 | true |
| val | `[20, 58]` | 58 | true |
| test | `[20, 58]` | 58 | true |

每个 split 输出：

- `physics_features.npz`：包含 `features [N,F]`、`feature_names [F]`、`sample_indices [N]`。
- `physics_features.csv`。
- `feature_summary.md`。

## Feature categories

- per-channel basic stats：`mean`、`std`、`min`、`max`、`peak_abs`、`peak_to_peak`。
- peak positions：`argmax_x`、`argmin_x`、`argmax_abs_x`。
- signal energy / area：`energy`、`abs_area`、`signed_area`。
- width / distribution：`positive_peak_count`、`negative_peak_count`、`half_abs_width`、`center_of_abs_mass`、`left_right_abs_balance`。
- lift-off decay ratios：`peak_abs_ch1_over_ch0`、`peak_abs_ch2_over_ch0`、`energy_ch1_over_ch0`、`energy_ch2_over_ch0`。
- inter-channel correlation：`corr_ch0_ch1`、`corr_ch0_ch2`、`corr_ch1_ch2`。

## 主要范围

| feature | train mean | val mean | test mean |
| --- | ---: | ---: | ---: |
| `ch0_peak_abs` | 4.862327e-04 | 5.021758e-04 | 4.841560e-04 |
| `ch0_half_abs_width` | 2.065126e-02 | 1.714573e-02 | 2.076382e-02 |
| `peak_abs_ch1_over_ch0` | 9.411271e-01 | 9.240862e-01 | 9.479756e-01 |
| `peak_abs_ch2_over_ch0` | 8.473197e-01 | 7.966241e-01 | 8.500682e-01 |
| `energy_ch1_over_ch0` | 6.990064e-01 | 6.868063e-01 | 7.031458e-01 |
| `energy_ch2_over_ch0` | 5.490007e-01 | 5.190657e-01 | 5.559245e-01 |
| `corr_ch0_ch1` | 9.826041e-01 | 9.824635e-01 | 9.806083e-01 |
| `corr_ch0_ch2` | 9.561715e-01 | 9.506714e-01 | 9.557550e-01 |
| `corr_ch1_ch2` | 9.703373e-01 | 9.641574e-01 | 9.698678e-01 |

Lift-off decay ratios 和 inter-channel correlations 在三个 split 中均为 finite，且大致符合多高度信号随 channel 变化的衰减/相关性直觉。

## 下一步

S142 将这些 features 作为 normalized auxiliary input 接入 parametric inverse runner，比较 `features_only` 与 `concat_latent`。

## 自评

- features 已成功生成，shape 与 feature_names 一致。
- feature 语义是轻量 physics-inspired diagnostics，不声称替代 COMSOL forward model。
- 当前没有发现 NaN/Inf 或 sample 对齐问题。
