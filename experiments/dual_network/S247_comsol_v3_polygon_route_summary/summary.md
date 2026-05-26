# S247 COMSOL V3 polygon geometry route summary

本阶段进入 polygon / corner-point route，但只做 schema、COMSOL smoke、hard rasterizer 和 oracle gate；不训练、不生成 larger pack、不替换当前 S185/S181 center-bin candidate。

## Why center/axis/rotation is insufficient

V3 raw COMSOL coordinates use `[0,4500] / [0,3000]`, while the V2-compatible branch uses `[-0.04,0.04] / [-0.01,0.01]`. This transform is non-uniform: `x_scale = 0.08 / 4500`, `y_scale = 0.02 / 3000`. A true rotated rectangle in raw space is therefore not generally representable after normalization by a single `center_x/y + axis_x/y + rotation_angle` tuple in the old schema.

The observed larger-pack oracle failure is therefore a representation ceiling, not a training issue. Continuing to force true V3 geometry through the old center/axis/rotation rasterizer would cap oracle IoU before the inverse model is involved.

## Route boundary

- Current branch candidate remains S185 `center_bin_offset_plus_grid` for V2-style parametric data.
- Polygon vertices become the proposed V3 true-geometry oracle representation.
- Legacy center/axis/rotation fields are retained only as auxiliary/debug fields for comparison.
- No dense runner, raster/forward consistency sweep, lambda sweep, or model training is run in this stage.

## Target direction

The first polygon target schema uses fixed four-corner polygons:

- `max_components = 3`
- `max_vertices = 4`
- `polygon_vertices_raw [N,3,4,2]`
- `polygon_vertices_norm [N,3,4,2]`
- `polygon_vertex_mask [N,3,4]`
- `presence_targets`, `type_targets`, `component_counts`, `sample_indices`, `type_vocab`

The training-facing representation is normalized vertices. Raw vertices are kept for COMSOL geometry audit.
