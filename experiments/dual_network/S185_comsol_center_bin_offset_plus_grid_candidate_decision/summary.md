# S185 COMSOL center-bin offset plus grid candidate decision

## Decision

Promote `center_bin_offset_plus_grid` to the current COMSOL parametric route candidate on `feature/dual-network-variational`.

This is a branch-local candidate, not a main baseline replacement.

## Current candidate

- raw MLP signal encoder
- shared parametric head
- fixed-order components
- `center_representation=bin_offset`
- `center_bin_size_cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no validation-aware endpoint selection

## Evidence

- S178 selected `center_bin_offset_plus_grid` in the 1500-step quick gate.
- S179 3000-step seed1 reached val / test IoU `0.542935` / `0.581320`.
- S183 added seed2 and seed3 with test IoU `0.575504` / `0.578738`.
- S184 showed all three runs exceed the S170 center-grid test IoU range and reduce held-out `center_grid_mae`.

## Stability caveat

Seed2 and seed3 pass the gate, but their val IoU is materially lower than S179 seed1. Therefore: 升级为支线当前 candidate，但仍需后续在 center-bin 下一阶段继续观察.

## Next step

Continue with the center-bin route, but do not tune lambda values blindly. The next stage should inspect why seed2/seed3 val IoU is lower than seed1, with attention to bin confidence, residual center error, and component-specific held-out failures. If later center-bin results become unstable, fall back to the S170 center-grid candidate and move to `signal-to-center auxiliary head`.

## Self-review

The decision upgrades the branch candidate because all acceptance criteria passed, while preserving the boundary that this is not final and not a main-branch baseline replacement.
