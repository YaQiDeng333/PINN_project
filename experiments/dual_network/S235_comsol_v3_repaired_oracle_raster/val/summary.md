# COMSOL parametric rasterization oracle summary

- npz_path: `experiments\dual_network\S233_comsol_v3_repaired_hard_case_ingest\converted\val_comsol_v3_repaired_hard_case.npz`
- parametric_targets: `experiments\dual_network\S234_comsol_v3_repaired_parametric_targets\val\parametric_targets.npz`
- axis semantics: `axis_x` / `axis_y` are treated as full width / full height.
- `rectangular_notch` and `rotated_rect` are both approximated as rotated rectangles.
- samples: `10`
- avg_oracle_iou: `1.000000e+00`
- min_oracle_iou: `1.000000e+00`
- max_oracle_iou: `1.000000e+00`
- avg_oracle_dice: `1.000000e+00`
- avg_target_area: `1.808000e+02`
- avg_raster_area: `1.808000e+02`
- avg_abs_area_diff: `0.000000e+00`

## Type sequences

- `rectangular_notch`: `10` samples

## Gate interpretation

- Gate passes when avg oracle IoU is at least 0.70 for train / val / test.
- Low oracle IoU would indicate target schema, axis semantics or rasterizer mismatch before any model training.
