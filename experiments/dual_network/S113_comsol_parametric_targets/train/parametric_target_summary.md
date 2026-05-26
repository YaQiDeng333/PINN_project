# COMSOL parametric target summary

- npz_path: `experiments\dual_network\S84_comsol_geometry_v2_data_ingest\converted\train_comsol_multiheight_v2.npz`
- raw defect rows: `100`
- samples: `100`
- max_components: `3`
- target_schema: `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- type_vocab: `rectangular_notch, rotated_rect`

## Presence count distribution

- `3` components: `100` samples

## Continuous target ranges

- `center_x`: min=-0.0202, max=0.0202, mean=-2.79397e-11
- `center_y`: min=-0.0057, max=0.005, mean=-0.000254667
- `axis_x`: min=0.004, max=0.0056, mean=0.00493333
- `axis_y`: min=0.005, max=0.0066, mean=0.005864
- `depth_or_shape_param`: min=0.001, max=0.0025, mean=0.00182917
- `rotation_angle`: min=-30, max=30, mean=-2.96667

## Notes

- Component sorting uses `center_x`, then `center_y`.
- `source_component_json` is used when available; sample-level fields are fallback only.
- No component truncation was applied; samples exceeding `max_components` raise `ValueError`.
