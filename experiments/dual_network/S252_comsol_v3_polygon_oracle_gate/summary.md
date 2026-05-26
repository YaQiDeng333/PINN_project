# S252 COMSOL V3 polygon oracle gate

S252 builds polygon targets from the S251 true COMSOL smoke and runs the hard polygon rasterizer against the smoke masks.

## Gate result

| sample | hard_case_type | polygon IoU | target area | raster area |
|---:|---|---:|---:|---:|
| 0 | high_angle_rotated_single | `1.000000` | `72` | `72` |
| 1 | two_component_union | `1.000000` | `97` | `97` |
| 2 | narrow_near_boundary_corner_safe | `1.000000` | `49` | `49` |

Mean IoU is `1.000000`; minimum IoU is `1.000000`. The gate threshold was per-sample IoU `>= 0.95`, so the polygon oracle gate passes.

## Interpretation

The true COMSOL smoke demonstrates that normalized polygon vertices can reconstruct masks after non-uniform coordinate normalization. This fixes the representation ceiling that blocked the old `center + axis + rotation` oracle on true rotated / multi-component V3 geometry.

This is still an oracle/data-representation gate, not an inverse-model training result.
