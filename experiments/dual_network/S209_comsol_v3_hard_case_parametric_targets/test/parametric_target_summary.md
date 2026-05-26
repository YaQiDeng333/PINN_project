# COMSOL parametric target summary

- npz_path: `experiments\dual_network\S208_comsol_v3_hard_case_ingest\converted\test_comsol_v3_hard_case.npz`
- raw defect rows: `10`
- samples: `10`
- max_components: `3`
- target_schema: `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- raw_target_schema: `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- angle_encoding: `raw`
- angle_unit: `degree`
- type_vocab: `rectangular_notch`

## Presence count distribution

- `1` components: `10` samples

## Continuous target ranges

- `center_x`: min=1066.33, max=3770.4, mean=2377.61
- `center_y`: min=1200, max=2340.91, mean=1651.21
- `axis_x`: min=420, max=850, mean=686
- `axis_y`: min=110, max=264, mean=190.9
- `depth_or_shape_param`: min=70, max=85, mean=76.5
- `rotation_angle`: min=0, max=0, mean=0

## Notes

- Component sorting uses `center_x`, then `center_y`.
- `source_component_json` is used when available; sample-level fields are fallback only.
- No component truncation was applied; samples exceeding `max_components` raise `ValueError`.
- `axis_x` / `axis_y` are component full width / full height values from `length_m` / `width_m` when `source_component_json` is available.
