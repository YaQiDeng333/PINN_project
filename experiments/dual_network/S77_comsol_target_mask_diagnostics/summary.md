# S77 COMSOL target / mask consistency diagnostics

## 目的

S77 诊断 S74 converted COMSOL geometry 数据中的 `mu_maps` 与 provided `masks` 是否一致，判断 S75 train IoU 偏低是否可能来自 mask label 构造差异。

## 输入数据

- train: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/train_comsol_multiheight.npz`
- val: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/val_comsol_multiheight.npz`
- test: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/test_comsol_multiheight.npz`

`mu_threshold = 500.0`，诊断比较：

- threshold mask: `mu_maps < 500.0`
- provided mask: `masks > 0.5`

## 结果

| split | samples | avg_threshold_area | avg_provided_mask_area | avg_abs_area_diff | avg_mask_iou | total_mismatch_count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 50 | 2344.18 | 2344.18 | 0.0 | 1.0 | 0 |
| val | 10 | 1988.90 | 1988.90 | 0.0 | 1.0 | 0 |
| test | 10 | 2398.40 | 2398.40 | 0.0 | 1.0 | 0 |

三个 split 均同时包含 `mu_maps` 和 `masks`。所有 `mu_maps` / `masks` 数值均为 finite。

## 判断

`mu_maps < 500.0` 与 provided `masks` 完全一致。S75 train IoU 偏低不是由 `mu_threshold` mask 与 provided masks 不一致导致。

当前建议：默认 `mask_source=mu_threshold` 与 `mask_source=masks` 等价；仍可通过 S78 运行 runner 路径对照，确认实现上没有隐藏差异。
