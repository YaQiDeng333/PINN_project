# True 3D RBC Real Data Intake Schema

Stage: 20.97 real-data schema intake / acquisition metadata contract.

Scope: this document defines the metadata and array contract required before real experimental MFL data can be passed into the current true 3D RBC inference path. It does not define a new baseline and does not change `CURRENT_BASELINE.md`.

Current inference route:

`delta_b + sensor_z_m -> 20.85 baseline or 20.85 baseline + A2 adapter -> L_m, W_m, D_m, wLD, wWD, wLW -> RBC-style profile/depth -> projected mask`

`sensor_z_m` is required. Nominal `0.008 m` routes to the frozen 20.85 baseline. Non-nominal values inside `[0.006, 0.012]` route to the A2 liftoff companion adapter. Missing `sensor_z_m` is a blocker. Out-of-range `sensor_z_m` must be flagged and is not validated for production inference.

## Format 1: Prepared Delta-B Format

This is the recommended intake format.

Required array:

- `delta_b`: shape `(N, 3, 3, 201)` for a batch or `(3, 3, 201)` for one sample.

Required metadata:

- `axis_order`: must be `["Bx", "By", "Bz"]`.
- `scan_line_y_m`: must map to three scan lines, recommended `[-0.001, 0.0, 0.001]`.
- `sensor_x_m`: length `201`, ordered along the scan direction.
- `sensor_z_m`: liftoff in meters, required per sample.
- `delta_b_unit`: must be `Tesla`.
- `sample_id`: required per sample.
- `specimen_id`: required per sample.
- `no_defect_reference_id`: required per sample.
- `coordinate_system`: required; must define scan direction, y-line convention, and liftoff direction.
- `no_defect_reference_method`: required; should describe matched reference acquisition.
- `sensor_alignment_status`: required; must state whether Bx/By/Bz are spatially aligned.
- `gain_calibration_status`: required; calibration is diagnostic only and does not replace the baseline.
- `material`: required when known.
- `specimen_info`: required when known.
- `magnetization_setup`: required.

Optional metadata:

- `split_tag`: optional reporting split tag, never a model input.
- `acquisition_date`: optional.
- `operator`: optional.
- `ground_truth_LWD`: optional but recommended for evaluation.
- `profile_depth_ground_truth`: optional.

## Format 2: Raw Defect Plus No-Defect Format

Use this when the raw defect scan and the matched no-defect reference are supplied separately.

Required arrays:

- `b_defect`: shape `(N, 3, 3, 201)` or `(3, 3, 201)`.
- `b_no_defect`: shape `(N, 3, 3, 201)` or `(3, 3, 201)`.
- `delta_b` must be computed as `b_defect - b_no_defect` before inference.

All metadata requirements from Format 1 still apply. `no_defect_reference_id` and `no_defect_reference_method` are mandatory.

## Blockers

These conditions stop real-data inference until fixed:

- Missing `sensor_z_m`.
- Missing no-defect reference when `delta_b` cannot be trusted.
- Only `Bz` is available; the current true 3D RBC model requires `Bx`, `By`, and `Bz`.
- Unknown axis order.
- Unknown magnetic field unit.
- `sensor_x_m` cannot be resampled to length `201`.
- `scan_line_y_m` cannot be mapped to exactly three scan lines.
- `sensor_z_m` is outside `[0.006, 0.012]` and there is no explicit out-of-range retraining or validation plan.
- Internal or buried defects are mixed into this surface-breaking RBC-style schema.

## Warnings

These do not automatically stop intake, but they must be reported:

- Gain or amplitude calibration is unknown.
- Sensor alignment is not verified.
- No ground-truth L/W/D or profile/depth target is available; inference may run but cannot be scored.
- Real specimen material or magnetization setup differs from the COMSOL simulation domain.

## Real-Data Inference Boundary

The current branch remains `exact_piao_rbc=False` and `rbc_style_approximation=True`. It is validated on COMSOL-derived surface-breaking RBC-style defects, not on arbitrary free-form, buried, internal, or multi-defect real data.
