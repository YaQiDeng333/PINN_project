# S157 oracle ablation split summary: test

## Aggregate IoU

| variant | avg_mask_iou | delta_vs_pred_all | oracle_gap | avg_abs_area_diff |
| --- | ---: | ---: | ---: | ---: |
| pred_all | 4.244624e-01 | 0.000000e+00 | 2.921214e-01 | 8.015000e+01 |
| gt_type | 4.244624e-01 | 0.000000e+00 | 2.921214e-01 | 8.015000e+01 |
| gt_rotation | 4.242472e-01 | -2.151978e-04 | 2.923366e-01 | 8.015000e+01 |
| gt_type_rotation | 4.242472e-01 | -2.151978e-04 | 2.923366e-01 | 8.015000e+01 |
| gt_center | 7.229199e-01 | 2.984575e-01 | -6.336106e-03 | 7.085000e+01 |
| gt_axis | 4.326793e-01 | 8.216901e-03 | 2.839045e-01 | 2.220000e+01 |
| gt_depth | 4.244624e-01 | 0.000000e+00 | 2.921214e-01 | 8.015000e+01 |
| gt_continuous_all | 7.165838e-01 | 2.921214e-01 | 0.000000e+00 | 1.405000e+01 |
| gt_type_continuous | 7.165838e-01 | 2.921214e-01 | 0.000000e+00 | 1.405000e+01 |
| gt_all | 7.165838e-01 | 2.921214e-01 | 0.000000e+00 | 1.405000e+01 |

## Interpretation

- `pred_all` IoU: `4.244624e-01`.
- `gt_all` IoU: `7.165838e-01`; 接近 S117 oracle gate。
- 单项替换最大提升: `gt_center`，delta `2.984575e-01`。
- 非 full-oracle 最大提升: `gt_center`，delta `2.984575e-01`。
- `gt_type` delta: `0.000000e+00`。当前 hard rasterizer 将 `rectangular_notch` 和 `rotated_rect` 都近似为 rotated rectangle；因此单独替换 type 通常不会改变 mask。
- `gt_rotation` delta: `-2.151978e-04`。

## Self-review

- 本脚本只读取已有 predictions / targets / masks，不训练模型，不保存权重、checkpoint 或图片。
- rotation 按 raw degree 语义处理，并在 rasterization 前转为 sin/cos schema，避免 hard rasterizer 的 degree/radian heuristic。
