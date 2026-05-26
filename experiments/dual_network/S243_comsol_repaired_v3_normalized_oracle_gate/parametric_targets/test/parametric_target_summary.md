# COMSOL parametric target summary

- npz_path: `experiments\dual_network\S242_comsol_repaired_v3_normalized_ingest\converted\test_comsol_v3_repaired_normalized_hard_case.npz`
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

- `center_x`: min=-0.0199908, max=0.0270294, mean=0.00216592
- `center_y`: min=-0.002, max=0.00532576, mean=0.000980051
- `axis_x`: min=0.00746667, max=0.0149333, mean=0.0114578
- `axis_y`: min=0.000733333, max=0.00173333, mean=0.00118333
- `depth_or_shape_param`: min=70, max=85, mean=76.5
- `rotation_angle`: min=0, max=0, mean=0

## Notes

- Component sorting uses `center_x`, then `center_y`.
- `source_component_json` is used when available; sample-level fields are fallback only.
- No component truncation was applied; samples exceeding `max_components` raise `ValueError`.
- `axis_x` / `axis_y` are component full width / full height values from `length_m` / `width_m` when `source_component_json` is available.
