# NLS Full-Compatible Feature Schema

Schema version: `nls_full_compatible_v1`

This schema defines a Piao NLS-full-compatible feature interface. It is a compatibility framework, not an exact reproduction of Piao full NLS. The current status is:

- `piao_full_compatible=true`
- `exact_piao_full=false`
- `exact_piao_full` may become `true` only after the full ROI input and the exact feature equations are validated against the source method.

## Required Input

The extractor consumes three aligned ROI matrices per sample:

- `Bx(M,K)`
- `By(M,K)`
- `Bz(M,K)`

The batched project convention is `delta_b` with shape `(S,3,M,K)`, where `S` is sample count, axis order is `[Bx, By, Bz]`, `M` is tangential scan-line count, and `K` is axial sensor-x count.

Required metadata:

- `axis_names=["Bx","By","Bz"]`
- `sensor_x_m` with length `K`
- `scan_line_y_m` with length `M`
- sample identifiers are optional for synthetic use but required for persisted dataset outputs.

Labels, split, sample_id, curvature templates, masks, depth bins, aspect bins, and model predictions are not feature inputs.

## Required Axes

The full-compatible extractor treats the ROI matrix as a two-axis field:

- axial axis = `x`, along each sensor line.
- tangential axis = `y`, across scan lines.

Axial local and decay features are extracted from each line or from the center/nearest-center line. Tangential envelope features require enough `y` samples to fit an envelope across scan lines.

## Scan-Line Adequacy Gate

`scan_line_count=M` controls the mode:

- `M < 5`: degraded-compatible mode only. `full_feature_ready=false`.
- `M >= 5`: full-compatible tangential envelope fitting may be attempted.
- `M >= 9`: full-candidate mode may be marked, subject to axis order, sensor-x count, y-line spacing, missing-value, and fit-feasibility checks.

The minimum tangential line count for full mode is `M >= 5`. The recommended full mode count is `M >= 9`.

Current v3_240 has `M=3` with `scan_line_y=[-0.001,0.0,0.001]`, so it must be marked degraded. It must not be described as exact Piao full NLS.

## Output Feature Groups

The extractor emits a fixed feature-name set across degraded, compatible, and full-candidate modes. Missing or inapplicable values are `NaN` and must have invalid validity flags.

Feature groups:

- `axial_local_features`: center-line peak, peak position, signed extrema, width, and energy.
- `axial_decay_features`: axial gradient, gradient energy, left/right asymmetry, and center-to-outer decay ratios.
- `tangential_envelope_features`: tangential envelope peak, peak position, width, Gaussian envelope parameters, and fit success.
- `cross_axis_features`: Bx/By/Bz peak ratios, correlations, and vector-magnitude features.
- `fit_residuals`: tangential Gaussian RMSE and normalized residual diagnostics.
- `feature_validity_flags`: one `valid__<feature_name>` column for every emitted feature column.

## Failure Policy

Fit failure must not be silent. The extractor records:

- numeric fit outputs as `NaN`;
- `valid__<fit_parameter>=false`;
- fit-success features as `0`;
- per-row `fit_failure_reasons`;
- aggregate quality rows with attempted, successful, and failed fit counts.

Downstream models must use train-only imputation and scaling if they consume this feature table. This framework does not train models.

## Current Status

The framework is compatible with the Piao 18-feature structure at the interface level because it separates axial, tangential, cross-axis, residual, and validity groups over three-axis ROI matrices. It is not exact Piao full NLS because current surface RBC v3_240 lacks full tangential ROI coverage and the exact equations have not been validated.
