# S117 COMSOL parametric rasterization oracle

## 目的

S117 检查 GT parametric targets 是否能通过非可微 rasterizer 重建 S84 target masks。该 gate 用来确认 parametric route 的上限：如果真实参数本身 rasterize 后 IoU 很低，则训练模型没有意义，应先修 rasterizer 或 target schema。

## Rasterizer 语义

- `axis_x` / `axis_y` 按 `source_component_json` 中 `length_m` / `width_m` 的 full width / full height 使用。
- rasterizer 内部使用 half width / half height 判断点是否落入 component。
- `rectangular_notch` 和 `rotated_rect` 第一版都近似为 rotated rectangle。
- 多 component mask 使用 union。

## Oracle 结果

| split | avg_oracle_iou | min_oracle_iou | max_oracle_iou | avg_oracle_dice | avg_target_area | avg_raster_area | avg_abs_area_diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 7.229967e-01 | 5.551048e-01 | 8.992740e-01 | 8.365438e-01 | 1.071170e+03 | 1.069190e+03 | 1.338000e+01 |
| val | 7.232882e-01 | 5.563910e-01 | 8.992740e-01 | 8.366408e-01 | 1.076750e+03 | 1.072250e+03 | 1.330000e+01 |
| test | 7.165838e-01 | 5.757576e-01 | 8.879781e-01 | 8.323042e-01 | 1.070000e+03 | 1.071050e+03 | 1.405000e+01 |

## Gate 判断

train / val / test 的平均 oracle IoU 均高于 `0.70`，因此 S117 oracle gate 通过。当前 parametric target + rasterizer 有足够上限，可以继续 S118 target refinement 和 S119 training probe。

## 仍需注意

- 部分样本最低 oracle IoU 约 `0.55`，说明 rectangular approximation 与真实 target mask 仍有局部差异。
- area 误差很小，主要差异更可能来自边界离散化、旋转矩形近似或 component 细节，而不是整体面积错配。
