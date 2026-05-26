# S294 Center-Anchored Polygon Held-Out Failure Summary

S292 showed that the center-anchored polygon runner has passed the train-fit gates but has not solved held-out generalization.

## Evidence

- One-sample gate IoU: `1.000000`.
- Five-sample gate mean/min IoU: `0.991549` / `0.957746`.
- Train30 gate mean/min IoU: `0.989276` / `0.857143`.
- Train30 presence/type accuracy: `1.000000` / `1.000000`.
- Train30 center x/y bin accuracy: `1.000000` / `1.000000`.
- Held-out val/test mean IoU: `0.072402` / `0.084416`.
- Held-out val/test zero-IoU samples: `8/10` / `8/10`.
- Held-out signed-area flips and out-of-grid vertices: `0` / `0`.

## Boundary

This stage does not run new training, does not modify the center-anchored model or runner, does not enter multi-seed validation, and does not replace the S185/S181 center-bin candidate. The goal is only to identify whether held-out failure is driven by center-bin coverage, x/y bin prediction, local shape regression, hard-case type, component slot, or true rotated / multi-component samples.
