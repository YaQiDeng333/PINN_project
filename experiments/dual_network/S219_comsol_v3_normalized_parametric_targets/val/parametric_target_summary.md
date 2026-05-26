# COMSOL parametric target summary

- npz_path: `experiments\dual_network\S218_comsol_v3_geometry_normalized\converted\val_comsol_v3_hard_case_normalized.npz`
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

- `center_x`: min=-0.0261083, max=0.0207661, mean=-0.00384683
- `center_y`: min=-0.0046, max=0.00237374, mean=-0.000868283
- `axis_x`: min=0.008, max=0.0156444, mean=0.0123733
- `axis_y`: min=0.0008, max=0.00173333, mean=0.001276
- `depth_or_shape_param`: min=70, max=85, mean=78.5
- `rotation_angle`: min=0, max=0, mean=0

## Notes

- Component sorting uses `center_x`, then `center_y`.
- `source_component_json` is used when available; sample-level fields are fallback only.
- No component truncation was applied; samples exceeding `max_components` raise `ValueError`.
- `axis_x` / `axis_y` are component full width / full height values from `length_m` / `width_m` when `source_component_json` is available.
