# S152 forward residual diagnostic (val)

| geometry_variant | avg_signal_nrmse | avg_signal_corr | avg_peak_abs_error |
| --- | ---: | ---: | ---: |
| axis_scaled_geometry | 2.661547e-01 | 9.577585e-01 | 3.956061e-05 |
| predicted_geometry | 7.572894e-01 | 7.595906e-01 | 8.703992e-05 |
| rotation_perturbed_geometry | 2.741137e+00 | 2.128940e-01 | 6.907084e-04 |
| true_geometry | 2.680198e-01 | 9.572266e-01 | 4.035870e-05 |
| type_swapped_geometry | 7.348850e-01 | 7.968278e-01 | 8.885960e-05 |

## 判断

- `type_swapped_geometry` 相对 true_geometry 的 avg_signal_nrmse delta = `4.668652e-01`。
- `rotation_perturbed_geometry` 相对 true_geometry 的 avg_signal_nrmse delta = `2.473117e+00`。
- `axis_scaled_geometry` 相对 true_geometry 的 avg_signal_nrmse delta = `-1.865078e-03`。
- `predicted_geometry` 相对 true_geometry 的 avg_signal_nrmse delta = `4.892696e-01`。
- predicted_geometry 的 signal_nrmse 与 pred_mask_iou 相关系数 = `-2.503890e-01`。

如果扰动 geometry 的 residual 没有明显高于 true_geometry，forward residual 不适合作为强训练 loss；如果只对部分扰动敏感，应作为 diagnostic 或定向约束使用。
