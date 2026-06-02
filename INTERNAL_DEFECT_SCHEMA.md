# Internal / Buried Defect Feasibility Schema

Stage: 20.99 internal / buried defect feasibility schema for true 3D MFL.

This document defines an independent internal-defect branch. It does not replace or update `CURRENT_BASELINE.md`. The current baseline remains the 20.85/20.86 surface or near-surface true 3D RBC profile-depth baseline, with `delta_b` Bx/By/Bz input and six RBC-style output parameters.

## Boundary

Surface / near-surface defect = a defect that intersects the scan surface or is close enough to be represented as a surface profile/depth field. The current surface RBC baseline represents this branch with `L_m`, `W_m`, `D_m`, `wLD`, `wWD`, and `wLW`, then generates a surface profile/depth grid and projected mask.

Internal / buried defect = a cavity or material discontinuity that does not break the scan surface. Its MFL response depends on cavity geometry, burial depth, center position, material path, liftoff, and sensor alignment. `D_m` is not a surface depth label in this branch; it is a cavity dimension. The branch must define `burial_depth_m` or `depth_to_surface_m` explicitly.

Internal / buried defects must not be mixed into the current surface RBC schema. The semantic target is volumetric or buried cavity geometry, not a surface pit profile.

## Required Inputs

The main internal branch requires:

- `Bx/By/Bz`: tri-axis MFL signal. Shape may reuse `(N, 3, 3, 201)` for the first smoke pack, but axis and scan-line metadata must be explicit.
- `Bz-only branch`: allowed only as a degraded diagnostic branch. It cannot be called the true 3D internal mainline and cannot replace tri-axis `Bx/By/Bz`.
- `sensor_z_m`: sensor liftoff from the scan surface, in meters.
- `b_defect`, `b_no_defect`, or a validated `delta_b`: a matched no-defect reference is required to compute `delta_b = b_defect - b_no_defect`.
- `no_defect_reference_id`: identifier for the matching reference scan or COMSOL run.
- `no_defect_reference_method`: how the reference was acquired or generated and how it was aligned to the defect run.
- `axis_order`: explicit channel order, recommended `["Bx", "By", "Bz"]`.
- `scan_line_y_m`: y positions of scan lines, recommended `[-0.001, 0.0, 0.001]` for the first smoke pack.
- `sensor_x_m`: x positions of scan samples, recommended 201 ordered samples for compatibility with current Conv1D tooling.
- `units`: magnetic field units convertible to `Tesla`; geometry in meters.
- `material`: material name or source for magnetic properties.
- `specimen_geometry`: block length, width, thickness, scan surface definition, and steel interior direction.
- `coordinate_system`: x scan direction, y transverse direction, z liftoff/depth direction, origin, and sign convention.
- `sensor_alignment_status`: whether Bx/By/Bz are spatially aligned.
- `gain_calibration_status`: whether channel gain/amplitude calibration is known.
- `magnetization_setup`: source current/magnetization direction and magnitude when known.

## Required Labels

Internal supervised training, quantitative benchmarking, or a COMSOL smoke pack must provide:

- `L_m`: cavity length in the x or major scan direction.
- `W_m`: cavity width in the y or transverse direction.
- `D_m` or `cavity_size_m`: cavity dimension in the z direction. This is not surface pit depth.
- `burial_depth_m`: distance from scan surface to the nearest cavity surface, in meters.
- `depth_to_surface_m`: equivalent to `burial_depth_m` unless the manifest defines a different reference point.
- `defect_center_xyz_m`: cavity center in the declared coordinate system.
- `shape_type`: for example `internal_ellipsoid`, `internal_cuboid`, or `sphere_like`.
- `profile_descriptor` or `cavity_mask`: parametric descriptor, voxel/occupancy mask, or projected equivalent used to describe cavity geometry.
- `ground_truth_method`: source of truth, for example COMSOL parameter table, machining plan, CT, sectioning, CAD, or measurement log.

Optional but recommended labels:

- `rotation_angle_rad` or orientation matrix for non-axis-aligned cavities.
- `size_level` and `burial_depth_level` for stratified smoke or pilot coverage.
- `specimen_id`, `scan_id`, and `reference_scan_id` for traceability.

## Output Representation Candidates

Recommended order for early work:

1. `shape_type + L/W/D + burial_depth + center_xyz`
   - Minimal smoke-pack representation.
   - Best first target for 6-12 samples.
   - Covers sizing, center, and burial without claiming free-form geometry.

2. `internal_ellipsoid_params`
   - Center, three axes, burial depth, and orientation.
   - Suitable for sphere-like and ellipsoid cavities.

3. `internal_cuboid_params`
   - Center, three dimensions, burial depth, and orientation.
   - Suitable for machined box-like cavities.

4. `3D occupancy / cavity mask`
   - Stronger representation for arbitrary internal cavities.
   - Requires a declared voxel grid, more data, and stronger validation.

5. `surface_equivalent_projected_profile`
   - A QA or comparator view only.
   - It should not be the sole target because it discards burial-depth semantics.

## Hard Blockers

The internal supervised branch is blocked if any of these are true:

- No `burial_depth_m` or `depth_to_surface_m`.
- No matched no-defect reference or no reliable `delta_b`.
- Only `Bz` is available and no explicit low-capability Bz-only route is declared.
- Defect location relative to the scan surface is unknown.
- Coordinate system or z sign convention is unknown.
- No ground truth, or ground-truth source is not documented.
- Missing `sensor_z_m`.
- Unknown material, specimen geometry, magnetization setup, sensor alignment, or gain status.

## Why Current Surface RBC Baseline Is Not Enough

Current surface RBC route:

`delta_b + sensor_z_m -> six RBC-style surface parameters -> surface profile/depth -> projected mask`

Internal defect route:

`delta_b + sensor_z_m + specimen/material context -> buried cavity geometry + burial_depth + center_xyz`

The current six RBC-style parameters have no `burial_depth_m` field and encode a surface profile, not a volumetric cavity. If internal samples are forced into that output space, the model can confuse burial-depth changes with surface profile depth, curvature, or footprint changes. That would produce physically wrong outputs even when the signal is finite.

Therefore 20.99 only authorizes schema, label, and data-generation design. It does not authorize COMSOL execution, training, data/NPZ generation, or a `CURRENT_BASELINE.md` update.
