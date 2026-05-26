# S158 COMSOL parametric oracle ablation results

## 目的

S158 在 S115 / S126 raw MLP baseline predictions 上运行 parameter-level oracle ablation，用已有预测逐项替换 GT 参数并重新 rasterize mask，不运行新训练。

## IoU 汇总

| split | pred_all | gt_type | gt_rotation | gt_type_rotation | gt_center | gt_axis | gt_depth | gt_continuous_all | gt_type_continuous | gt_all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 6.980716e-01 | 6.980716e-01 | 6.984192e-01 | 6.984192e-01 | 7.233959e-01 | 6.984778e-01 | 6.980716e-01 | 7.229967e-01 | 7.229967e-01 | 7.229967e-01 |
| val | 3.699078e-01 | 3.699078e-01 | 3.695472e-01 | 3.695472e-01 | 7.148715e-01 | 3.810309e-01 | 3.699078e-01 | 7.232882e-01 | 7.232882e-01 | 7.232882e-01 |
| test | 4.244624e-01 | 4.244624e-01 | 4.242472e-01 | 4.242472e-01 | 7.229199e-01 | 4.326793e-01 | 4.244624e-01 | 7.165838e-01 | 7.165838e-01 | 7.165838e-01 |

## Delta vs pred_all

| split | gt_type | gt_rotation | gt_type_rotation | gt_center | gt_axis | gt_depth | gt_continuous_all | gt_type_continuous | gt_all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.000000e+00 | 3.476511e-04 | 3.476511e-04 | 2.532431e-02 | 4.062734e-04 | 0.000000e+00 | 2.492512e-02 | 2.492512e-02 | 2.492512e-02 |
| val | 0.000000e+00 | -3.606208e-04 | -3.606208e-04 | 3.449637e-01 | 1.112311e-02 | 0.000000e+00 | 3.533803e-01 | 3.533803e-01 | 3.533803e-01 |
| test | 0.000000e+00 | -2.151978e-04 | -2.151978e-04 | 2.984575e-01 | 8.216901e-03 | 0.000000e+00 | 2.921214e-01 | 2.921214e-01 | 2.921214e-01 |

## 与 S117 oracle 对齐

`gt_all` 与 S117 oracle 一致：

- train: `7.229967e-01`
- val: `7.232882e-01`
- test: `7.165838e-01`

因此 prediction CSV / S113 targets / hard rasterizer 的对齐可信。`gt_center` 在 train / test 略高于 `gt_all`，说明 approximate rasterizer 下预测的 axis / rotation 有时会补偿 GT rasterizer gap；这不是对齐失败，因为 `gt_all` 已精确复现 S117 oracle。

## 主要结论

- `gt_center` 是单项替换中最大、最稳定的 IoU 提升来源：val 提升 `3.449637e-01`，test 提升 `2.984575e-01`。
- `gt_axis` 有小幅提升：val `1.112311e-02`，test `8.216901e-03`。
- `gt_rotation` 没有改善 val/test，反而略降。
- `gt_type` 对 mask IoU 没有影响，因为当前 hard rasterizer 将 `rectangular_notch` 和 `rotated_rect` 都近似为 rotated rectangle。
- `gt_depth` 没有影响，因为当前 rasterizer 不使用 depth 生成 mask。
- `gt_continuous_all` 与 `gt_all` 一致，presence / type 不是当前 mask IoU gap 的直接限制。

## 自评

- train / val / test 均成功运行。
- 没有训练、checkpoint、权重或图片输出。
- 结论稳定：主要瓶颈是 `center_x` / `center_y` localization，而不是 type / rotation loss。
