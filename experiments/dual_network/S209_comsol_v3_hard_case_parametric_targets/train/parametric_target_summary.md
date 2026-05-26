# COMSOL parametric target summary

- npz_path: `experiments\dual_network\S208_comsol_v3_hard_case_ingest\converted\train_comsol_v3_hard_case.npz`
- raw defect rows: `30`
- samples: `30`
- max_components: `3`
- target_schema: `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- raw_target_schema: `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- angle_encoding: `raw`
- angle_unit: `degree`
- type_vocab: `rectangular_notch`

## Presence count distribution

- `1` components: `30` samples

## Continuous target ranges

- `center_x`: min=704.07, max=3770.4, mean=2200.27
- `center_y`: min=810, max=2342.42, mean=1476.76
- `axis_x`: min=420, max=880, mean=692.667
- `axis_y`: min=110, max=264, mean=192.8
- `depth_or_shape_param`: min=70, max=85, mean=77.1667
- `rotation_angle`: min=0, max=0, mean=0

## Notes

- Component sorting uses `center_x`, then `center_y`.
- `source_component_json` is used when available; sample-level fields are fallback only.
- No component truncation was applied; samples exceeding `max_components` raise `ValueError`.
- `axis_x` / `axis_y` are component full width / full height values from `length_m` / `width_m` when `source_component_json` is available.
