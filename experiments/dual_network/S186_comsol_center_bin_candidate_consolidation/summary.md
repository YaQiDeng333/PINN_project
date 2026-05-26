# S186 COMSOL center-bin candidate consolidation

## Why S185 replaces the S170 center-grid candidate

S170 promoted raw MLP / shared head / fixed-order + `lambda_center_grid=0.1` because it was the first stable center-targeted improvement. S181-S185 then validated the structured `center_bin_offset_plus_grid` variant across three runs. All three center-bin runs beat the S170 center-grid test IoU range while keeping lower held-out center error.

## Stability results

| run | val IoU | test IoU | val center_grid_mae | test center_grid_mae |
|---|---:|---:|---:|---:|
| seed1 reused S179 | 0.542935 | 0.581320 | 3.362513 | 2.721649 |
| seed2 | 0.484303 | 0.575504 | 6.282760 | 2.929023 |
| seed3 | 0.492127 | 0.578738 | 6.026593 | 2.804331 |

## Current COMSOL parametric candidate

- raw MLP
- shared head
- fixed-order
- `center_representation=bin_offset`
- `center_bin_size_cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no validation-aware endpoint selection

## Boundary

- This is the current candidate only on `feature/dual-network-variational`.
- It is not a main baseline replacement.
- It is not a final solution.
- Seed2/seed3 have lower val IoU than S179 seed1, so later center-bin stages should keep observing validation stability.

## Self-review

S186 only consolidates existing S181-S185 evidence. No new training, Python change, checkpoint, weight, image, or `.npy` artifact was created.
