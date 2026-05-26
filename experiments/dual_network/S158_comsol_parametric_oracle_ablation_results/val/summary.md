# S157 oracle ablation split summary: val

## Aggregate IoU

| variant | avg_mask_iou | delta_vs_pred_all | oracle_gap | avg_abs_area_diff |
| --- | ---: | ---: | ---: | ---: |
| pred_all | 3.699078e-01 | 0.000000e+00 | 3.533803e-01 | 1.055000e+02 |
| gt_type | 3.699078e-01 | 0.000000e+00 | 3.533803e-01 | 1.055000e+02 |
| gt_rotation | 3.695472e-01 | -3.606208e-04 | 3.537410e-01 | 1.068500e+02 |
| gt_type_rotation | 3.695472e-01 | -3.606208e-04 | 3.537410e-01 | 1.068500e+02 |
| gt_center | 7.148715e-01 | 3.449637e-01 | 8.416645e-03 | 1.057000e+02 |
| gt_axis | 3.810309e-01 | 1.112311e-02 | 3.422572e-01 | 1.195000e+01 |
| gt_depth | 3.699078e-01 | 0.000000e+00 | 3.533803e-01 | 1.055000e+02 |
| gt_continuous_all | 7.232882e-01 | 3.533803e-01 | 0.000000e+00 | 1.330000e+01 |
| gt_type_continuous | 7.232882e-01 | 3.533803e-01 | 0.000000e+00 | 1.330000e+01 |
| gt_all | 7.232882e-01 | 3.533803e-01 | 0.000000e+00 | 1.330000e+01 |

## Interpretation

- `pred_all` IoU: `3.699078e-01`.
- `gt_all` IoU: `7.232882e-01`; 接近 S117 oracle gate。
- 单项替换最大提升: `gt_center`，delta `3.449637e-01`。
- 非 full-oracle 最大提升: `gt_continuous_all`，delta `3.533803e-01`。
- `gt_type` delta: `0.000000e+00`。当前 hard rasterizer 将 `rectangular_notch` 和 `rotated_rect` 都近似为 rotated rectangle；因此单独替换 type 通常不会改变 mask。
- `gt_rotation` delta: `-3.606208e-04`。

## Self-review

- 本脚本只读取已有 predictions / targets / masks，不训练模型，不保存权重、checkpoint 或图片。
- rotation 按 raw degree 语义处理，并在 rasterization 前转为 sin/cos schema，避免 hard rasterizer 的 degree/radian heuristic。
