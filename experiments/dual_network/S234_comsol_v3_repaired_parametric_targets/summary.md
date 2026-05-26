# S234 COMSOL V3 repaired parametric targets

S234 rebuilds fixed-order parametric targets from S233 converted NPZ files and repaired V3 `defect_params.csv`.

## Target Schema

- `max_components = 3`
- `target_schema = center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- `raw_target_schema = center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- `angle_encoding = raw`
- `angle_unit = degree`
- `type_vocab = rectangular_notch`
- presence distribution: all samples contain one component
- component sorting: `center_x`, then `center_y`

## Outputs

- `experiments/dual_network/S234_comsol_v3_repaired_parametric_targets/train/parametric_targets.npz`
- `experiments/dual_network/S234_comsol_v3_repaired_parametric_targets/val/parametric_targets.npz`
- `experiments/dual_network/S234_comsol_v3_repaired_parametric_targets/test/parametric_targets.npz`

No component truncation was applied.
