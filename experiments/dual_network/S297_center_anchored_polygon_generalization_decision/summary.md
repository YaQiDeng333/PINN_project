# S297 Center-Anchored Polygon Generalization Decision

S294-S297 diagnose the held-out failure without running training or changing the runner.

## S295 Failure Source

The failure is center-bin dominated, with y-bin as the clearest bottleneck. Across val/test, all `16/16` zero-IoU samples have at least one center-bin error, all `16/16` have y-bin errors, and `8/16` have x-bin errors. Held-out split y-bin accuracy is much weaker than x-bin accuracy: val `0.200000` vs `0.500000`, test `0.100000` vs `0.750000`.

Local shape regression is secondary in this run. Present components with correct held-out bins have mean `local_vertex_grid_mae=0.867101`; components with wrong bins have mean `2.487230`. Only `2/20` held-out samples have all center bins correct, and their mean IoU is `0.468456`, so local-shape repair alone is not the first move.

The hardest held-out hard-case label is `both_bins_wrong_like` with mean IoU `0.000000`; component slot `1` is the weakest present slot with mean IoU `0.000000`. True rotated and true multi-component samples are harder than their counterparts: rotated zero-IoU rate is `0.933333`, and multi-component zero-IoU rate is `1.000000`.

## S296 Coverage Source

The held-out failures correlate strongly with train center-bin coverage gaps. There are `19` uncovered held-out component bins; `15/16` zero-IoU samples have at least one uncovered bin, while only `1` nonzero-IoU sample has an uncovered bin. Zero-IoU samples also have larger nearest-train center-bin distance than nonzero samples: `1.468750` vs `0.250000`.

## Decision

The next unique recommendation is a matched-coverage resplit gate using the existing polygon V3 pack. That should test whether held-out zero-IoU collapses when train/val/test share center-bin coverage before adding model complexity, extra steps, multi-seed validation, or a larger COMSOL pack.

If the matched-coverage resplit still fails with covered bins, the next repair should target y-bin localization and local-shape conditioning. If the matched-coverage resplit succeeds, the current `30/10/10` split is too sparse for held-out candidate validation, and the follow-up should be coverage-balanced V3 data design.

The S185/S181 center-bin candidate remains unchanged, and this stage is not a main baseline replacement.
