# S344 Component-Query Boundary Precision Raster Recheck

S344 reruns offline raster-sensitivity diagnostics for all three S343 outputs.

The best actual prediction is `center_aux_half`: IoU `0.989528796`, FP/FN `2 / 0`, symmetric diff `2`, and pred/target area `191 / 189`. The sensitivity variants show that a small additional alignment would cross the gate: `pred_center_gt_local_vertices`, `gt_center_pred_local_vertices`, area-scaled, centroid-aligned, edge-scaled, and alpha-interpolated variants each reach at least IoU `0.994736842`.

This confirms the model is close to the hard-raster threshold, but the actual trained output still does not meet acceptance. The correct stage decision is to stop at 1-sample, not to infer that 5-sample is safe.
