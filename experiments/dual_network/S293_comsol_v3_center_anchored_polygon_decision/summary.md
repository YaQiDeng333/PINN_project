# S293 COMSOL V3 Center-Anchored Polygon Decision

S289-S293 validates the center-anchored polygon representation through the staged train gates.

## Result

- Center/local decode oracle passes with train/val/test mean and min IoU `1.000000`.
- One-sample gate passes with IoU `1.000000`.
- Five-sample gate passes with mean/min IoU `0.991549` / `0.957746`.
- Train30 gate passes with mean/min IoU `0.989276` / `0.857143`.
- The weakest train hard-case group is `rare_y_bin_wrong` with mean/min IoU `0.974117` / `0.956835`.

## Held-Out Interpretation

Center anchoring improves the geometric pathology seen in S284-S288: this run has `0` signed-area flips and `0` out-of-grid vertices on val/test. However, held-out IoU remains low: val/test mean IoU is `0.072402` / `0.084416`, with zero-IoU counts `8` / `8`.

The remaining bottleneck is now center-bin/local-shape generalization, not absolute-vertex coordinate blow-up. Val/test center-bin accuracy is weak, especially y-bin accuracy: val `0.461538` / `0.153846`, test `0.769231` / `0.076923`.

## Decision

The center-anchored polygon route is worth continuing because it passes all train-fit gates and removes the main out-of-grid / signed-area pathology. It is not yet a validated held-out candidate. Do not enter multi-seed validation or candidate replacement.

Next recommended direction: diagnose and repair held-out center-bin localization for the center-anchored polygon runner, likely with controlled resplit / matched-coverage diagnostics before adding model complexity.

The S185/S181 center-bin candidate remains unchanged, and the absolute-vertex polygon runner remains the S282 reference.
