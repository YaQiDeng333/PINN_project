# True 3D RBC Liftoff-Conditioned Inference Metadata Contract

Stage: 20.96 liftoff-conditioned inference smoke

Scope: current true 3D RBC nominal baseline plus the A2 liftoff robustness companion module. This is not a CURRENT_BASELINE replacement.

## Required Input Fields

- `delta_b`: defect-minus-no-defect magnetic flux leakage signal.
- `sensor_z_m`: sensor liftoff in meters. This field is mandatory; inference must not guess it.
- `sample_id`: optional identifier for reporting only, never a model input.

## Signal Shape And Axis Order

- Raw shape: `(N, 3, 3, 201)`.
- Axes: `Bx`, `By`, `Bz` in that order.
- Scan-line axis: three `scan_line_y` rows.
- Conv1D model shape: `(N, 9, 201)` after flattening `(axis, scan_line_y)`.

## Liftoff Metadata

- Unit: meters.
- Nominal liftoff: `sensor_z_m = 0.008`.
- Supported validated range: `[0.006, 0.012]`.
- Nominal route tolerance: `abs(sensor_z_m - 0.008) < 0.0005`.
- Missing `sensor_z_m`: hard error.
- Out-of-range `sensor_z_m`: inference may return a prediction, but the result must be flagged `out_of_range` and must not be treated as validated.

## Routing

- `auto`: use frozen 20.85 baseline at nominal liftoff; use baseline plus A2 adapter at non-nominal liftoff.
- `force_baseline`: use frozen 20.85 baseline for all rows, for comparison only.
- `force_adapter`: use baseline plus A2 adapter for all rows, for comparison only.
- Output must include `route_used`, either `baseline` or `baseline_plus_adapter`.

## Required Preprocessing

- `delta_b` must be computed as `b_defect - b_no_defect` with the no-defect reference acquired under matching geometry and liftoff where possible.
- Channel order and units must match the COMSOL-derived v3_240/liftoff pack convention.
- No target labels, projected masks, profile grids, split labels, bins, or sample identifiers may be used as model inputs.

## Outputs

- RBC-style six parameters: `L_m`, `W_m`, `D_m`, `wLD`, `wWD`, `wLW`.
- Generated `profile_depth_grid`.
- Generated projected mask.
- Metrics, when labels exist: profile depth RMSE, Er-like profile error, L/W/D MAE, wMAE auxiliary, projected mask IoU/Dice.

## Caveats

- `exact_piao_rbc=False`; this branch remains an RBC-style / Piao-inspired approximation.
- A2 is a liftoff robustness companion module, not the CURRENT_BASELINE.
- Internal or buried defects remain outside this inference contract.
