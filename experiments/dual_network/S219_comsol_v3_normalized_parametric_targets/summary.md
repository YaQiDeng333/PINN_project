# S219 COMSOL V3 normalized parametric targets and oracle gate

S219 rebuilds parametric targets from the S218 normalized V3 geometry and reruns oracle rasterization on the normalized grids.

## Target Schema

- `presence_targets`: `[N,3]`
- `type_targets`: `[N,3]`
- `continuous_targets`: `[N,3,6]`
- `target_schema`: `center_x`, `center_y`, `axis_x`, `axis_y`, `depth_or_shape_param`, `rotation_angle`
- `type_vocab`: `rectangular_notch`

Depth values are retained from raw V3 because depth is not used by the current hard mask rasterizer.

## Oracle Gate

| split | samples | avg oracle IoU | min oracle IoU | avg Dice | avg abs area diff |
|---|---:|---:|---:|---:|---:|
| train | 30 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| val | 10 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| test | 10 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

## Decision

The normalized target/grid convention passes the oracle gate. It is safe to proceed to S220 runability gate.
