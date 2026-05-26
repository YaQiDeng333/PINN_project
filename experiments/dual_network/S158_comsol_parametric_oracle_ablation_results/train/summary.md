# S157 oracle ablation split summary: train

## Aggregate IoU

| variant | avg_mask_iou | delta_vs_pred_all | oracle_gap | avg_abs_area_diff |
| --- | ---: | ---: | ---: | ---: |
| pred_all | 6.980716e-01 | 0.000000e+00 | 2.492512e-02 | 1.224000e+01 |
| gt_type | 6.980716e-01 | 0.000000e+00 | 2.492512e-02 | 1.224000e+01 |
| gt_rotation | 6.984192e-01 | 3.476511e-04 | 2.457747e-02 | 1.245000e+01 |
| gt_type_rotation | 6.984192e-01 | 3.476511e-04 | 2.457747e-02 | 1.245000e+01 |
| gt_center | 7.233959e-01 | 2.532431e-02 | -3.991899e-04 | 1.355000e+01 |
| gt_axis | 6.984778e-01 | 4.062734e-04 | 2.451885e-02 | 1.219000e+01 |
| gt_depth | 6.980716e-01 | 0.000000e+00 | 2.492512e-02 | 1.224000e+01 |
| gt_continuous_all | 7.229967e-01 | 2.492512e-02 | 0.000000e+00 | 1.338000e+01 |
| gt_type_continuous | 7.229967e-01 | 2.492512e-02 | 0.000000e+00 | 1.338000e+01 |
| gt_all | 7.229967e-01 | 2.492512e-02 | 0.000000e+00 | 1.338000e+01 |

## Interpretation

- `pred_all` IoU: `6.980716e-01`.
- `gt_all` IoU: `7.229967e-01`; 接近 S117 oracle gate。
- 单项替换最大提升: `gt_center`，delta `2.532431e-02`。
- 非 full-oracle 最大提升: `gt_center`，delta `2.532431e-02`。
- `gt_type` delta: `0.000000e+00`。当前 hard rasterizer 将 `rectangular_notch` 和 `rotated_rect` 都近似为 rotated rectangle；因此单独替换 type 通常不会改变 mask。
- `gt_rotation` delta: `3.476511e-04`。

## Self-review

- 本脚本只读取已有 predictions / targets / masks，不训练模型，不保存权重、checkpoint 或图片。
- rotation 按 raw degree 语义处理，并在 rasterization 前转为 sin/cos schema，避免 hard rasterizer 的 degree/radian heuristic。
