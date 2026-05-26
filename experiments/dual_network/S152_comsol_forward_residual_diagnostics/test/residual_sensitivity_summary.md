# S152 forward residual diagnostic (test)

| geometry_variant | avg_signal_nrmse | avg_signal_corr | avg_peak_abs_error |
| --- | ---: | ---: | ---: |
| axis_scaled_geometry | 2.691335e-01 | 9.568606e-01 | 3.975516e-05 |
| predicted_geometry | 7.149153e-01 | 7.132657e-01 | 5.347003e-05 |
| rotation_perturbed_geometry | 4.292691e+00 | 5.014552e-01 | 1.615126e-03 |
| true_geometry | 2.606982e-01 | 9.590282e-01 | 3.749634e-05 |
| type_swapped_geometry | 4.385888e-01 | 8.902319e-01 | 5.223715e-05 |

## 判断

- `type_swapped_geometry` 相对 true_geometry 的 avg_signal_nrmse delta = `1.778906e-01`。
- `rotation_perturbed_geometry` 相对 true_geometry 的 avg_signal_nrmse delta = `4.031993e+00`。
- `axis_scaled_geometry` 相对 true_geometry 的 avg_signal_nrmse delta = `8.435345e-03`。
- `predicted_geometry` 相对 true_geometry 的 avg_signal_nrmse delta = `4.542171e-01`。
- predicted_geometry 的 signal_nrmse 与 pred_mask_iou 相关系数 = `-4.378152e-01`。

如果扰动 geometry 的 residual 没有明显高于 true_geometry，forward residual 不适合作为强训练 loss；如果只对部分扰动敏感，应作为 diagnostic 或定向约束使用。
