# S334 Component-Query 1-Sample Raster Sensitivity Diagnostic

- pred IoU: `0.974227`
- pred Dice: `0.986945`
- pred / target area: `194` / `189`
- area diff: `5`
- false-positive / false-negative pixels: `5` / `0`
- symmetric diff pixels: `5`
- max vertex error: `0.039043` grid cells
- max edge length error: `0.032379` x-grid cells

## Findings

1. The `0.974227` IoU is primarily `false-positive` driven.
2. The 5-pixel raster area surplus matches the FP/FN balance.
3. Area-scaled IoU is `0.979275`, centroid-aligned IoU is `1.000000`, and edge-scaled IoU is `0.979275`.
4. Best alpha interpolation variant is `pred_polygon_interpolate_gt_alpha_0.75` with IoU `0.994737`.
5. Recommendation: 1-sample repair quick gate with the corresponding centroid/center auxiliary; do not enter 5-sample yet.

## Variant Table

| variant | IoU | pred area | area diff | FP | FN | sym diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pred_polygon | `0.974227` | `194` | `5` | `5` | `0` | `5` |
| gt_polygon | `1.000000` | `189` | `0` | `0` | `0` | `0` |
| pred_center_gt_local_vertices | `0.979275` | `193` | `4` | `4` | `0` | `4` |
| gt_center_pred_local_vertices | `1.000000` | `189` | `0` | `0` | `0` | `0` |
| pred_polygon_area_scaled_to_target | `0.979275` | `193` | `4` | `4` | `0` | `4` |
| pred_polygon_centroid_aligned_to_target | `1.000000` | `189` | `0` | `0` | `0` | `0` |
| pred_polygon_edge_length_scaled_to_target | `0.979275` | `193` | `4` | `4` | `0` | `4` |
| pred_polygon_interpolate_gt_alpha_0.25 | `0.984375` | `192` | `3` | `3` | `0` | `3` |
| pred_polygon_interpolate_gt_alpha_0.50 | `0.989529` | `191` | `2` | `2` | `0` | `2` |
| pred_polygon_interpolate_gt_alpha_0.75 | `0.994737` | `190` | `1` | `1` | `0` | `1` |
