# S246 repaired COMSOL V3 normalized evaluation decision

S242-S246 fixes the repaired V3 coordinate convention and reruns oracle, zero-shot, and train quick-probe evaluation. No new pack was generated, the training runner and model structure were not changed, and no main baseline replacement is claimed.

## Findings

1. Repaired V3 normalization succeeded: train/val/test converted shapes remain `[30,3,200]`, `[10,3,200]`, and `[10,3,200]`; normalized `x/y` ranges match the V2-compatible `[-0.04,0.04] / [-0.01,0.01]` convention.
2. Oracle rasterization remains perfect after normalization: train/val/test IoU is `1.000000` / `1.000000` / `1.000000`.
3. V2-trained zero-shot now runs without center-bin range errors, but val/test IoU is only `0.007616` / `0.005248`.
4. Normalized repaired V3 train is learnable: the current candidate reaches train IoU `1.000000`.
5. Held-out normalized repaired V3 remains weak: candidate val/test IoU is `0.055172` / `0.188341`; param-only reference val/test IoU is `0.000000` / `0.171178`.
6. Grouped diagnostics show broad held-out failure, with y-bin weakness and hard-case split sensitivity still visible.

## Current Candidate Boundary

The current COMSOL parametric route candidate remains the S185 center-bin configuration for the branch:

- raw MLP / shared head / fixed-order
- `center_representation=bin_offset`
- `center_bin_size_cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- no raster loss
- no forward consistency
- no val selection

It is still branch-local and not a main baseline replacement.

## Next Recommendation

The next unique recommendation is to generate a larger repaired V3 hard-case pack before mixed V2+V3 training or candidate promotion. The current `30/10/10` fallback pilot proves that repaired signals and normalized targets are valid, but it is too small and split-sensitive to support a stable held-out candidate decision.
