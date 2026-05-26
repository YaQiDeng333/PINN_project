# COMSOL target / mask consistency diagnostics

- npz_path: `experiments\dual_network\S74_comsol_geometry_data_ingest\converted\test_comsol_multiheight.npz`
- mu_threshold: `500.0`
- has_mu_maps: `True`
- has_masks: `True`
- samples: `10`

## Aggregate metrics

- samples: `10`
- avg_threshold_area: `2.398400e+03`
- avg_provided_mask_area: `2.398400e+03`
- avg_abs_area_diff: `0.000000e+00`
- avg_mask_iou: `1.000000e+00`
- min_mask_iou: `1.000000e+00`
- max_mask_iou: `1.000000e+00`
- total_mismatch_count: `0`

`mu_maps < mu_threshold` 与 provided `masks` 完全一致。
当前建议：`mask_source=mu_threshold` 和 `mask_source=masks` 等价，可继续使用默认 `mu_threshold`。