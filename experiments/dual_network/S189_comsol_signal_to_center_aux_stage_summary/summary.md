# S189 COMSOL signal-to-center auxiliary head stage summary

## Context

S186-S188 consolidated `center_bin_offset_plus_grid` as the current COMSOL parametric route candidate on `feature/dual-network-variational`. That candidate remains branch-local and is not a main baseline replacement.

The S187 diagnostics showed:

- test behavior is stable because test `center_grid_mae` stays near `2.72-2.93`;
- validation behavior is less stable because seed2/seed3 val `center_grid_mae` rises above `6.0`;
- the remaining bottleneck appears to be x-bin stability first, with y-bin stability secondary.

## Stage Direction

S189-S193 tests a signal-to-center auxiliary head as a structural diagnostic for center-bin stability. The stage does not tune center-bin lambda values, does not return to raster/forward/type/rotation losses, and does not use the dense conditional mask runner.

The current candidate remains the S185 configuration:

- raw MLP;
- shared head;
- fixed-order components;
- `center_representation=bin_offset`;
- `center_bin_size_cells=8`;
- `lambda_center_bin=1.0`;
- `lambda_center_offset=1.0`;
- `lambda_center_grid=0.1`;
- `lambda_center_axis_relative=0.0`;
- no raster loss;
- no forward consistency;
- no validation-aware endpoint selection.

## Self-Review

- The stage is targeted at the residual center-bin stability issue identified in S187.
- It preserves the current candidate boundary and does not claim a main baseline replacement.
- It uses a same-round quick-gate reference so auxiliary results are not judged only against historical runs.
