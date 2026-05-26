# S345 Component-Query Boundary Precision Decision

S341-S345 does not pass the component-query 1-sample boundary precision gate.

Acceptance required hard IoU `>=0.99`, pred/target area gap `<=2`, FP+FN below current reference, presence/type/bin accuracy `1.0`, no out-of-grid vertices, no signed-area flip, and no vertex-MAE regression. `center_aux_half` satisfies every condition except hard IoU: it reaches `0.989528796`, with area gap `+2`, FP/FN `2 / 0`, no out-of-grid vertices, and no signed-area flip.

Because all three actual trained outputs remain below IoU `0.99`, the 5-sample and train30 gates remain blocked.

Next unique recommendation: do not expand this into a loss sweep. Plan a targeted boundary-aware 1-sample repair that directly optimizes the remaining two hard-raster boundary pixels, or explicitly revisit whether the `>=0.99` one-sample gate is too strict for this small 189-pixel mask.
