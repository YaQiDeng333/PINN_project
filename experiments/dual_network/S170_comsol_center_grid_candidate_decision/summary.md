# S170 COMSOL center-grid candidate decision

## Decision

Promote raw MLP / shared head / fixed-order + `lambda_center_grid=0.1` to the current COMSOL parametric route candidate.

## Evidence

- S158 showed `gt_center` nearly closes the baseline-to-oracle mask IoU gap.
- S161 showed val/test center error strongly correlates with low mask IoU.
- S163 quick gate showed `center_grid_loss` improves val/test against same-round 1500-step reference.
- S164 3000-step full probe improved val/test over historical param-only baseline.
- S168/S169 stability repeat preserved improvement across `existing_unrecorded`, seed1, and seed2.

## Stability result

| run_id | val IoU | test IoU | val center_grid_mae | test center_grid_mae |
|---|---:|---:|---:|---:|
| existing_unrecorded | 0.469423 | 0.498874 | 5.996350 | 5.546025 |
| center_grid_seed1 | 0.485716 | 0.505590 | 5.443171 | 4.931658 |
| center_grid_seed2 | 0.446966 | 0.503713 | 6.732050 | 4.872537 |

## Current best configuration

- `encoder_type=mlp`
- `head_mode=shared`
- `component_matching_mode=fixed`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no val selection

## Next step

Use this candidate as the reference for the next center-localization stage. Do not continue lambda sweeps. If further gains are needed, move to center representation changes such as center-bin classification + offset, signal-to-center auxiliary head, or per-component peak-position alignment.

## Boundary

This is the current best candidate on the dual-network COMSOL parametric branch. It is not a main-branch replacement claim and should still be validated with future route-specific probes.
