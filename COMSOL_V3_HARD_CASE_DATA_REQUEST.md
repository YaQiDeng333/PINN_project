# COMSOL V3 Hard-Case Data Request

## Purpose

Create a small V2-compatible COMSOL V3 hard-case pilot for the `feature/dual-network-variational` COMSOL parametric route. This request targets the mixed failure modes found in S204; it is not a broad V3 expansion, not a training result, and not a main baseline replacement.

The current branch candidate remains S185 `center_bin_offset_plus_grid`: raw MLP / shared head / fixed-order, `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, `lambda_center_grid=0.1`, no auxiliary head, no raster loss, no forward consistency, and no validation selection.

## Requested Size

Default hard-case pilot:

| split | samples |
| --- | ---: |
| train | 60 |
| val | 20 |
| test | 20 |

If COMSOL generation cost is too high, use fallback:

| split | samples |
| --- | ---: |
| train | 30 |
| val | 10 |
| test | 10 |

Confirm generation budget before asking COMSOL to run the default `60/20/20` pack.

## Hard-Case Mix

Default `60/20/20` mix:

| hard_case_label | train | val | test | intent |
| --- | ---: | ---: | ---: | --- |
| `x_bin_wrong_like` | 24 | 8 | 8 | Stress x-bin boundary and slot 0 / slot 2 center localization. |
| `both_bins_wrong_like` | 9 | 3 | 3 | Stress coupled x/y ambiguity and near component spacing. |
| `bins_correct_center_or_offset_bad` | 15 | 5 | 5 | Keep bins nominally correct while stressing offset magnitude, decoded center, axis, and geometry interaction. |
| `geometry_or_type_interaction` | 9 | 3 | 3 | Stress mixed type sequences, rotated rectangles, and mask geometry interaction. |
| `rare_y_bin_wrong` | 3 | 1 | 1 | Preserve a small y-bin hard-case slice without making it the main direction. |

Fallback `30/10/10` mix:

| hard_case_label | train | val | test |
| --- | ---: | ---: | ---: |
| `x_bin_wrong_like` | 12 | 4 | 4 |
| `both_bins_wrong_like` | 5 | 1 | 2 |
| `bins_correct_center_or_offset_bad` | 8 | 2 | 2 |
| `geometry_or_type_interaction` | 4 | 2 | 1 |
| `rare_y_bin_wrong` | 1 | 1 | 1 |

## Geometry Design Rules

- Place centers near `center_bin_size_cells=8` x/y bin boundaries, with extra density near x boundaries.
- Include slot 0 and slot 2 hard cases after fixed-order component sorting by `center_x`, then `center_y`.
- Include near and medium component distances to create component interference without changing the fixed `max_components=3` setup.
- Include mixed `rectangular_notch` / `rotated_rect` type sequences and high-angle rotated rectangles.
- Include small-area and narrow components near target-area ranges that previously showed higher center-grid error.
- For `bins_correct_center_or_offset_bad`, do not simply put centers in the wrong bin. Keep the intended bin stable and stress offset magnitude, decoded center, axis, rotation, type sequence, and mask geometry interaction.

## Required Split Structure

Each split directory must contain:

- `signals_multiheight.csv`
- `targets.npz`
- `defect_params.csv`
- `README.md`

Train / val / test must have non-overlapping `source_sample_id` and `source_global_index` values. Val/test should contain the same hard-case categories as train, but not duplicate train geometry points.

## `signals_multiheight.csv`

The signal CSV must remain compatible with `convert_comsol_multiheight_csv_to_npz.py`.

Required columns:

- `sample_index`
- `channel_index`
- `channel_name`
- `lift_off`
- `field_component`
- `x_index`
- `x`
- `value`

Required values:

- `channel_index`: `0`, `1`, `2`
- `channel_name`: `Bz_liftoff_0p5`, `Bz_liftoff_1p0`, `Bz_liftoff_2p0`
- `lift_off_values`: `[0.5, 1.0, 2.0]`
- `field_component`: `Bz`
- `signal_len`: `200`
- rows per split: `num_samples * 3 * 200`

## `targets.npz`

Required arrays / metadata:

- `mu_maps [N,100,200] float32`
- `masks [N,100,200] float32`
- `x [200]`
- `y [100]`
- `defect_params` structured array, or a fully aligned same-directory `defect_params.csv`
- `source_sample_ids`
- `source_global_indices`
- `signal_channel_names`
- `lift_off_values`
- `field_components`
- `source_type`
- `signal_flatten_order=channels_first`
- `geometry_units`
- `field_units`
- `metadata_json`

## `defect_params.csv`

Keep the V2 fields:

- `sample_index`, `split`, `source_sample_id`, `source_global_index`
- `defect_type`
- `defect_center_x`, `defect_center_y`, `defect_center_z`
- `defect_axis_x`, `defect_axis_y`, `defect_axis_z`
- `defect_radius_or_width`
- `defect_depth_or_shape_param`
- `defect_mu`
- `rotation_angle`
- `boundary_irregularity`
- `boundary_irregularity_level`
- `c_magn`, `mur_magn`, `Mr_magn_A_per_m`
- `component_count`
- `component_types`
- `distance_bin`
- `min_pairwise_component_distance`
- `source_component_json`

Each component inside `source_component_json` must include:

- `component_id`
- `component_type`
- `center_x_m`
- `center_y_m`
- `center_z_m`
- `length_m`
- `width_m`
- `depth_m`
- `angle_deg`
- `angle_rad`

Optional diagnostic columns may be appended without breaking V2 compatibility:

- `hard_case_label`
- `hard_case_source`
- `design_note`
- `bin_boundary_axis`
- `target_slot_focus`

## Ingest Gate

After generation, do not train immediately. First run the ingest gate:

1. Convert train / val / test with `convert_comsol_multiheight_csv_to_npz.py`.
2. Validate converted NPZ files with the COMSOL multi-height validator.
3. Build parametric targets with `comsol_parametric_targets.py`.
4. Confirm sample order is split-local `0..N-1`.
5. Confirm `signals` shape is `[N,3,200]`, `x` length is `200`, and `y` length is `100`.
6. Confirm no NaN / Inf in signals or targets.
7. Confirm every split covers the five planned `hard_case_label` categories.

## Anti-Copy Rules

- Do not directly copy S84 / V2 samples.
- Use S204 hard samples only as failure templates.
- Apply controlled perturbations for bin-boundary distance, component spacing, type sequence, rotation, axis, and area.
- Keep train / val / test source ids disjoint.
- Do not generate ordinary random V2-like samples and label them as V3 hard cases.

## Acceptance

The data request is accepted when:

- the pack is V2-compatible;
- converter and validator pass;
- `defect_params.csv` and `targets.npz` can rebuild S113-style parametric targets;
- every split covers all five hard-case categories with the requested counts;
- split README files state that this is a hard-case pilot, not a broad V3 dataset and not a candidate replacement;
- no training result is claimed.

## Stop Conditions

- Stop if COMSOL cannot provide complete V2-compatible schema fields.
- Stop if `source_component_json` cannot represent fixed-order-compatible components.
- Use fallback `30/10/10` if default `60/20/20` is too expensive.
- Stop if only ordinary V2-like random cases can be generated.
- Stop and add decoded-center / offset / geometry-interaction diagnostics first if bins-correct low-IoU mechanisms remain too unclear to design cases.
