# S161 COMSOL parametric center diagnostics summary

## 诊断对象

使用 S126 `s115_raw_mlp_export` predictions、S113 raw parametric targets、S84 V2 converted NPZ 和 S126 per-sample mask metrics，对 train / val / test 的 per-component center error 做 grid-cell、axis-relative 和 mask-IoU correlation 诊断。

## Split-level metrics

| split | center_x_grid_mae | center_y_grid_mae | center_l2_grid_mae | center_x_axis_relative_mae | center_y_axis_relative_mae | center_axis_relative_l2_mae | pred_mask_iou |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.801384 | 0.363953 | 0.940224 | 0.066130 | 0.012700 | 0.068994 | 0.698072 |
| val | 4.375079 | 5.998688 | 8.017750 | 0.364193 | 0.203189 | 0.445062 | 0.369908 |
| test | 3.832278 | 5.097501 | 6.998191 | 0.315701 | 0.176160 | 0.394215 | 0.424462 |

## Center error vs mask IoU correlation

| split | error | pearson | spearman |
|---|---|---:|---:|
| train | center_l2_grid_mae | -0.234355 | -0.174701 |
| train | center_axis_relative_l2_mae | -0.226910 | -0.158896 |
| val | center_l2_grid_mae | -0.926528 | -0.902256 |
| val | center_axis_relative_l2_mae | -0.972801 | -0.974436 |
| test | center_l2_grid_mae | -0.910617 | -0.908271 |
| test | center_axis_relative_l2_mae | -0.966971 | -0.959398 |

## 分组观察

- Held-out val/test 的 center grid error 明显大于 train；val/test 的 `center_y_grid_mae` 高于 `center_x_grid_mae`，说明 y direction localization 更弱。
- Val/test 的 center error 与 mask IoU 呈强负相关；axis-relative L2 与 IoU 的相关性略强于 grid L2。
- Slot 误差不是唯一解释：val 三个 slot 的 L2 grid error 都偏高，test 也没有单一 slot 独占问题。
- 该诊断与 S158 oracle ablation 一致：center localization 是当前 mask IoU gap 的直接解释。

## 推荐 loss

优先 quick-gate `grid_mse` 和 `axis_relative_smoothl1` 两类 center-specific objective。`axis_relative_smoothl1` 更贴近 component 尺度；`grid_mse` 更直接对齐 rasterized mask 的离散 grid sensitivity。S163 需以同一轮 1500-step reference 为主比较，不直接用 S115 3000-step 作为 gate baseline。

## 自评

S161 强制检查了 x/y grid、target schema、prediction CSV 字段和 center true 值对齐；未发现 y-axis flip、slot mismatch 或 grid spacing 风险。Center diagnostics 足以支持继续 S162/S163。
