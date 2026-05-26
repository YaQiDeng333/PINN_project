# S243 repaired COMSOL V3 normalized oracle gate

S243 rebuilds fixed-order parametric targets from the S242 normalized repaired V3 pack, then reruns hard oracle rasterization on the normalized grid.

## Target Schema

- `max_components=3`
- `type_vocab=rectangular_notch`
- continuous schema: `center_x`, `center_y`, `axis_x`, `axis_y`, `depth_or_shape_param`, `rotation_angle`
- Geometry convention: normalized V2-compatible `x/y`, `center_x/y`, and `axis_x/y`; raw depth/z retained from repaired V3.

## Oracle Gate

| split | samples | oracle IoU | min IoU | max IoU | Dice | avg target area | avg raster area | avg abs area diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 30 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 184.566667 | 184.566667 | 0.000000 |
| val | 10 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 180.800000 | 180.800000 | 0.000000 |
| test | 10 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 176.600000 | 176.600000 | 0.000000 |

The oracle gate passes. Normalization preserved target/raster alignment, so S244 can test V2-train to normalized repaired V3 runability and zero-shot metrics.
