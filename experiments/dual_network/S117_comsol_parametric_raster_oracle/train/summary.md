# COMSOL parametric rasterization oracle summary

- npz_path: `experiments\dual_network\S84_comsol_geometry_v2_data_ingest\converted\train_comsol_multiheight_v2.npz`
- parametric_targets: `experiments\dual_network\S113_comsol_parametric_targets\train\parametric_targets.npz`
- axis semantics: `axis_x` / `axis_y` are treated as full width / full height.
- `rectangular_notch` and `rotated_rect` are both approximated as rotated rectangles.
- samples: `100`
- avg_oracle_iou: `7.229967e-01`
- min_oracle_iou: `5.551048e-01`
- max_oracle_iou: `8.992740e-01`
- avg_oracle_dice: `8.365438e-01`
- avg_target_area: `1.071170e+03`
- avg_raster_area: `1.069190e+03`
- avg_abs_area_diff: `1.338000e+01`

## Type sequences

- `rectangular_notch|rectangular_notch|rectangular_notch`: `25` samples
- `rectangular_notch|rectangular_notch|rotated_rect`: `25` samples
- `rectangular_notch|rotated_rect|rotated_rect`: `25` samples
- `rotated_rect|rotated_rect|rotated_rect`: `25` samples

## Gate interpretation

- Gate passes when avg oracle IoU is at least 0.70 for train / val / test.
- Low oracle IoU would indicate target schema, axis semantics or rasterizer mismatch before any model training.
