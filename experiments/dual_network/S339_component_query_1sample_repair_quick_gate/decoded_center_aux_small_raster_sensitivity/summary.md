# S334 Component-Query 1-Sample Raster Sensitivity Diagnostic

- pred IoU: `0.984127`
- pred Dice: `0.992000`
- pred / target area: `186` / `189`
- area diff: `-3`
- false-positive / false-negative pixels: `0` / `3`
- symmetric diff pixels: `3`
- max vertex error: `0.081857` grid cells
- max edge length error: `0.112637` x-grid cells

## Findings

1. The `0.974227` IoU is primarily `false-negative` driven.
2. The 5-pixel raster area surplus matches the FP/FN balance.
3. Area-scaled IoU is `0.984127`, centroid-aligned IoU is `0.989529`, and edge-scaled IoU is `0.984127`.
4. Best alpha interpolation variant is `pred_polygon_interpolate_gt_alpha_0.75` with IoU `0.994709`.
5. Recommendation: Do not enter 5-sample; first repair component-query one-sample local-shape precision.

## Variant Table

| variant | IoU | pred area | area diff | FP | FN | sym diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pred_polygon | `0.984127` | `186` | `-3` | `0` | `3` | `3` |
| gt_polygon | `1.000000` | `189` | `0` | `0` | `0` | `0` |
| pred_center_gt_local_vertices | `0.984127` | `186` | `-3` | `0` | `3` | `3` |
| gt_center_pred_local_vertices | `0.989529` | `191` | `2` | `2` | `0` | `2` |
| pred_polygon_area_scaled_to_target | `0.984127` | `186` | `-3` | `0` | `3` | `3` |
| pred_polygon_centroid_aligned_to_target | `0.989529` | `191` | `2` | `2` | `0` | `2` |
| pred_polygon_edge_length_scaled_to_target | `0.984127` | `186` | `-3` | `0` | `3` | `3` |
| pred_polygon_interpolate_gt_alpha_0.25 | `0.984127` | `186` | `-3` | `0` | `3` | `3` |
| pred_polygon_interpolate_gt_alpha_0.50 | `0.989418` | `187` | `-2` | `0` | `2` | `2` |
| pred_polygon_interpolate_gt_alpha_0.75 | `0.994709` | `188` | `-1` | `0` | `1` | `1` |
