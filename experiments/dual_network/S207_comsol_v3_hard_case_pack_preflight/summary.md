# S207 COMSOL V3 hard-case pack preflight

## Decision

Generate a small V2-compatible COMSOL V3 hard-case pilot only after compute-budget confirmation. Do not train, do not replace the current candidate, and do not expand into a broad ordinary V3 dataset.

The current branch COMSOL parametric candidate remains S185 `center_bin_offset_plus_grid`: raw MLP / shared head / fixed-order, `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, `lambda_center_grid=0.1`, no auxiliary head, no raster loss, no forward consistency, and no validation selection.

## Pack Size

Default request:

- train = `60`
- val = `20`
- test = `20`

Fallback if COMSOL generation cost is high:

- train = `30`
- val = `10`
- test = `10`

## Hard-Case Mix

The default mix follows S204's mixed failure taxonomy rather than treating the residual issue as a pure x-bin problem:

| hard_case_label | train | val | test |
| --- | ---: | ---: | ---: |
| `x_bin_wrong_like` | 24 | 8 | 8 |
| `both_bins_wrong_like` | 9 | 3 | 3 |
| `bins_correct_center_or_offset_bad` | 15 | 5 | 5 |
| `geometry_or_type_interaction` | 9 | 3 | 3 |
| `rare_y_bin_wrong` | 3 | 1 | 1 |

The fallback `30/10/10` mix is documented in `COMSOL_V3_HARD_CASE_DATA_REQUEST.md`.

## Schema Boundary

The pack must stay V2-compatible: per split, COMSOL should provide `signals_multiheight.csv`, `targets.npz`, `defect_params.csv`, and `README.md`. Signals remain three Bz channels with length `200`, `lift_off_values=[0.5,1.0,2.0]`, target grid `100x200`, and component metadata compatible with `comsol_parametric_targets.py`.

## Ingest Before Training

After generation, the first branch step should be an ingest gate only:

1. convert `signals_multiheight.csv` + `targets.npz`;
2. validate converted NPZ schema;
3. build parametric targets;
4. confirm split-local sample order;
5. confirm hard-case label coverage.

No training should happen until this gate passes.

## Self-Review

- This is a data-design stage, not a model or loss change.
- It keeps the S185 candidate frozen and branch-local.
- It requires user / COMSOL-side cost confirmation before generation.
