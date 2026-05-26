# S211 COMSOL V3 hard-case ingest gate decision

S208-S210 completed the V3 hard-case ingest gate for the real COMSOL fallback pilot.

## Decision

The pack can proceed to V3 hard-case candidate evaluation. The immediate next evaluation should be the current S185 COMSOL parametric candidate on V3 hard-case val/test, first as a zero-shot diagnostic and only then, if justified, as a carefully scoped fine-tune experiment.

## Evidence

- train/val/test converted shapes: `[30,3,200]`, `[10,3,200]`, `[10,3,200]`
- CSV row counts: `18000`, `6000`, `6000`
- hard-case labels cover all requested fallback classes.
- values are finite, with no NaN/Inf.
- `masks` exactly match `mu_maps < 500`.
- parametric target schema is `center_x`, `center_y`, `axis_x`, `axis_y`, `depth_or_shape_param`, `rotation_angle`.
- oracle raster train/val/test IoU is `1.000000` / `1.000000` / `1.000000`.

## Boundary

This is a real COMSOL hard-case fallback pilot and is not synthetic or copied from V2. It currently covers single rectangular-notch Block solves only. It does not yet cover true rotated-rect or multi-component solved geometry, so any downstream result must keep that limitation explicit.

## Next Step

Run an ingest-only model evaluation plan for the current S185 `center_bin_offset_plus_grid` candidate on V3 hard-case val/test. Do not treat this pack as a broad V3 benchmark until rotated and multi-component COMSOL geometries are generated and pass the same ingest/oracle gates.
