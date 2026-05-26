# S232 COMSOL V3 repaired Bz signal 3-sample smoke

本阶段直接执行 3-sample 真实 COMSOL mini-smoke，用于验证 S231 后确定的 repaired Bz export 路线是否不只在单个样本上成立。本阶段不训练模型、不生成 fallback pack、不修改 dual-network runner，也不把本 smoke 写成候选模型性能结论。

## 输入和输出

- Java 入口：`ComsolV3BzSignalRepairMiniSmoke.java`
- 真实 COMSOL model：`magnetic_prospecting.mph`
- 输出目录：`comsol_geometry_v3_bz_signal_repair_3sample_smoke/`
- candidate signal：`near_defect delta_Bz_solve`
- schema 边界：`signals_multiheight.csv` 中 `field_component=Bz`，但数值是 repaired near-defect anomaly / delta-Bz signal，用于保持 V2-compatible shape。

三个样本：

| sample_index | hard_case_type | 设计重点 |
|---:|---|---|
| 0 | `x_bin_wrong_like` | center_x 靠近 x-bin 边界 |
| 1 | `bins_correct_center_or_offset_bad` | center offset 与 axis_x / axis_y 变化 |
| 2 | `rare_y_bin_wrong` | center_y / y-offset 变化 |

## Signal self-check

所有检查均通过：

- CSV rows = `1800`，等于 `3 * 3 * 200`。
- 每个 sample/channel 的 `x_index` 完整覆盖 `0..199`。
- `value` 无 NaN / Inf。
- 每个 sample/channel 的 signal std 均 `> 1e-8`。
- 每个 sample/channel 的 peak-to-peak 均 `> 1e-8`。
- 三个 lift-off channel 之间存在差异。
- 三个 sample 之间存在差异。
- `targets.npz` 包含 `mu_maps`、`masks`、`x`、`y`、`defect_params`。
- `masks == (mu_maps < 500)` mismatch = `0`。
- `converted_3sample_smoke.npz` shape = `[3,3,200]`。

Per-sample candidate signal summary：

| sample_index | hard_case_type | std_min | std_max | peak_to_peak_min | peak_to_peak_max |
|---:|---|---:|---:|---:|---:|
| 0 | `x_bin_wrong_like` | `1.002105e-06` | `1.013198e-06` | `4.505153e-06` | `4.544585e-06` |
| 1 | `bins_correct_center_or_offset_bad` | `9.849913e-07` | `1.046541e-06` | `7.369206e-06` | `7.829620e-06` |
| 2 | `rare_y_bin_wrong` | `1.037377e-06` | `1.054912e-06` | `4.972724e-06` | `5.056808e-06` |

## Decision

3-sample mini-smoke 成功。old probe / raw absolute Bz 的 near-constant failure 已经被 repaired near-defect anomaly / delta-Bz route 修复，且该路线在 x-bin、offset/axis、y-offset 三类 hard-case 参数上均产生非退化空间波形。

下一步建议：生成 repaired V3 hard-case fallback pack，但仍保持边界清楚：它应是下一阶段的数据生成任务，不是本阶段训练任务，也不能把本 smoke 当作 fallback pack 或模型性能结果。
