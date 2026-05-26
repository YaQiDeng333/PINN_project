# COMSOL parametric target summary

- npz_path: `experiments\dual_network\S218_comsol_v3_geometry_normalized\converted\train_comsol_v3_hard_case_normalized.npz`
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

- `center_x`: min=-0.0274832, max=0.0270294, mean=-0.000884124
- `center_y`: min=-0.0046, max=0.00561616, mean=-0.00015495
- `axis_x`: min=0.00746667, max=0.0156444, mean=0.0123141
- `axis_y`: min=0.000733333, max=0.00176, mean=0.00128533
- `depth_or_shape_param`: min=70, max=85, mean=77.1667
- `rotation_angle`: min=0, max=0, mean=0

## Notes

- Component sorting uses `center_x`, then `center_y`.
- `source_component_json` is used when available; sample-level fields are fallback only.
- No component truncation was applied; samples exceeding `max_components` raise `ValueError`.
- `axis_x` / `axis_y` are component full width / full height values from `length_m` / `width_m` when `source_component_json` is available.
