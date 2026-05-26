# COMSOL parametric target summary

- npz_path: `experiments\dual_network\S233_comsol_v3_repaired_hard_case_ingest\converted\val_comsol_v3_repaired_hard_case.npz`
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

- `center_x`: min=781.407, max=3359.69, mean=2031.88
- `center_y`: min=810, max=1856.06, mean=1369.76
- `axis_x`: min=450, max=880, mean=666.5
- `axis_y`: min=120, max=260, mean=180
- `depth_or_shape_param`: min=70, max=85, mean=78.5
- `rotation_angle`: min=0, max=0, mean=0

## Notes

- Component sorting uses `center_x`, then `center_y`.
- `source_component_json` is used when available; sample-level fields are fallback only.
- No component truncation was applied; samples exceeding `max_components` raise `ValueError`.
- `axis_x` / `axis_y` are component full width / full height values from `length_m` / `width_m` when `source_component_json` is available.
