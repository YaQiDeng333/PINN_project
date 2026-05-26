# S335 Component-Query 1-Sample Raster Sensitivity Decision

S334 confirms the S330 failure is a hard-raster boundary sensitivity issue driven by a very small centroid / center displacement.

## Key Evidence

- `pred_polygon` IoU: `0.974227`
- false-positive / false-negative pixels: `5` / `0`
- pred / target area: `194` / `189`
- symmetric diff pixels: `5`
- max vertex error: `0.039043` grid cells
- `pred_center + gt_local_vertices` IoU: `0.979275`
- `gt_center + pred_local_vertices` IoU: `1.000000`
- `pred_polygon_centroid_aligned_to_target` IoU: `1.000000`
- area-scaled and edge-scaled variants only reach `0.979275`
- alpha interpolation toward GT reaches `0.989529` at `0.50` and `0.994737` at `0.75`

## Decision

Do not enter 5-sample yet. The single-sample miss is not explained by local vertices alone: replacing the center with GT while keeping predicted local vertices reaches IoU `1.000000`, while keeping predicted center with GT local vertices remains below gate. The next stage should run a 1-sample repair quick gate focused on center / centroid precision, not train30 or held-out generalization.

## Next Recommendation

Add a tiny default-off component-query center / centroid auxiliary or equivalent one-sample precision repair, then rerun only the 1-sample gate. Area-only or edge-only auxiliary is less supported because area-scaled and edge-scaled variants did not reach `0.99`.

## Non-Actions

No new training was run in S333-S335. No 5-sample, no train30, no multi-seed, no new COMSOL data, no model promotion, no S185/S181 replacement, and no push.
