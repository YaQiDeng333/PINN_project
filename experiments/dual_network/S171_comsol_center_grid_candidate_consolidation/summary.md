# S171 COMSOL center-grid candidate consolidation

## Stage purpose

S171 consolidates the S166-S170 stability result into the current COMSOL parametric route decision. This stage is documentation-only: it adds no training, no new seed, no Python change, and no CLI default change.

## Evidence from S166-S170

- S168 reused the S164 full probe as `existing_unrecorded` and added seed1 / seed2 recorded runs.
- The three center-grid runs had val / test mask IoU:
  - `existing_unrecorded`: `0.469423` / `0.498874`
  - `center_grid_seed1`: `0.485716` / `0.505590`
  - `center_grid_seed2`: `0.446966` / `0.503713`
- All three runs exceeded the historical param-only baseline of val / test `0.369908` / `0.424462`.
- All three runs reduced val / test `center_grid_mae` relative to the S161 baseline.
- The improvement is not dependent on the unseeded S164 run.

## Consolidated candidate

The current COMSOL parametric route candidate on `feature/dual-network-variational` is:

- raw MLP signal encoder
- shared parametric head
- fixed-order component regression
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no validation-aware endpoint selection

## Boundary

This candidate is a branch-local COMSOL parametric route candidate. It is not a main baseline replacement, not a main-branch claim, and not a final unsupervised inversion result.

## Self-review

S171 accurately records the S166-S170 evidence and does not overstate the result. The next documentation step is to add a reproducible command and register the candidate in the branch docs.
