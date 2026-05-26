# S87 COMSOL V1/V2 target distribution diagnostics

## 核心结论

- train mean label area ratio: V1 = `1.172090e-01`, V2 = `5.355850e-02`, V2/V1 = `0.457`。
- V2 label area 与 V1 差异较大，可能影响 IoU 和训练拟合。
- V1/V2 的 `mu_maps < 500` 与 provided `masks > 0.5` 均完全一致；target/mask 定义未发现异常。

## aggregate target distribution

| dataset | split | samples | mean_label_area_ratio | avg_mask_iou | mu_min | mu_max | defect_mu_mean | background_mu_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | train | 50 | 1.172090e-01 | 1.000000e+00 | 1.000000e+00 | 1.000000e+03 | 1.000000e+00 | 1.000000e+03 |
| v1 | val | 10 | 9.944500e-02 | 1.000000e+00 | 1.000000e+00 | 1.000000e+03 | 1.000000e+00 | 1.000000e+03 |
| v1 | test | 10 | 1.199200e-01 | 1.000000e+00 | 1.000000e+00 | 1.000000e+03 | 1.000000e+00 | 1.000000e+03 |
| v2 | train | 100 | 5.355850e-02 | 1.000000e+00 | 1.000000e+00 | 1.000000e+03 | 1.000000e+00 | 1.000000e+03 |
| v2 | val | 20 | 5.383750e-02 | 1.000000e+00 | 1.000000e+00 | 1.000000e+03 | 1.000000e+00 | 1.000000e+03 |
| v2 | test | 20 | 5.350000e-02 | 1.000000e+00 | 1.000000e+00 | 1.000000e+03 | 1.000000e+00 | 1.000000e+03 |

## 当前判断

- V2 target/mask 本身未发现不一致异常。
- V2 defect distribution 比 V1 更复杂：V2 是 `rectangular_notch` / `rotated_rect` multi_defect 组合，并包含 rotation 与 boundary proxy；V1 是上一批 COMSOL geometry pilot。
- V2 低 IoU 更可能来自任务难度、signal 语义或 runner/loss 对 multi_defect 目标适配不足，而不是简单 mask label 错误。
