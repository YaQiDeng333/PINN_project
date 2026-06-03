# 25.19 Geometry-Primary Component-Set Plan

- route decision: `A. enter 25.20 separated/close two-component geometry-primary training gate only if continuing beyond current completion package; no baseline transition`
- source route reset: `STOP_RASTER_TARGET_MAINLINE`
- training run: `False`
- loss tuning: `False`
- model capacity expanded: `False`
- CURRENT_BASELINE updated: `False`
- missing 25.18 input files: `0`

## Main Conclusion

25.19 turns the stopped raster-target mainline into a geometry-primary component-set future-work package. The main output should be component geometry slots; mask/depth become geometry-derived evaluation and weak diagnostic artifacts, not the primary per-component raster loss.

## 25.18 Input Evidence

All requested 25.18 route-reset files were present:

- `results/metrics/25_18_raster_target_route_reset_metrics.json`
- `results/summaries/25_18_raster_target_route_reset_summary.md`
- `results/manifests/25_18_raster_target_route_reset_manifest.json`

The carried-forward decision is `STOP_RASTER_TARGET_MAINLINE`: label-v2 target-v2, label-v3 soft support, label-v3b hard-core/halo/SDF raster supervision, and loss rebalance / label-v4 raster-target tuning should not continue as the mainline.

## Geometry-Primary Schema

`K=3` means maximum component slots, not Piao kernels. A two-component sample has two existing slots and a third non-existing slot; the inactive slot is supervised only through non-existence and is masked out of real-component geometry losses.

Each slot predicts:

- `existence_prob`
- `center_x_m`
- `center_y_m`
- `L_m`
- `W_m`
- `D_m`
- `rotation_angle`
- `shape_family`
- `compact_shape_parameters`

Labels should come from explicit raw component geometry metadata, `component_params_json`, and component-level raw masks/depths only as diagnostic cross-checks. Missing geometry fields are recorded and blocked from supervised use rather than guessed.

## Geometry-Derived Targets And Evaluator

The geometry slots derive:

- `derived_component_mask`
- `derived_union_mask`
- `derived_component_depth`
- `derived_union_depth`

Per-component raw raster mask/depth no longer define the main supervision. Raw component masks/depths remain useful for evaluation, weak diagnostics, and label-derivation sanity checks. Union mask/depth are the final derived-quality check, and reporting must stay grouped by `separated`, `close`, `touching`, and `partially_overlapping`.

## Matching Cost

Hungarian matching should prioritize geometry identity:

- existence
- center distance
- `L/W/D` relative error
- rotation error
- optional `shape_family` consistency

Raster mask loss is not a primary matching cost. Empty slots need separate non-existence handling so they are not treated as missed real components. Near-circular or weak-rotation samples may use rotation uncertainty or confidence-aware angular scoring.

## 25.20 First Gate

The first follow-up gate is `25.20 separated/close two-component geometry-primary training gate`. It uses only separated / close two-component samples and excludes touching, partially overlapping, and three-component samples.

Gate metrics:

- component-count accuracy
- component recall
- missed / extra / merged rate
- center error
- `L/W/D` relative error
- rotation error
- derived union Dice / IoU
- derived depth RMSE

PASS means geometry beats degenerate predictors, merged rate does not return to `1.0`, recall does not collapse, and derived union mask/depth are reasonable. FAIL means separated/close cases still cannot be separated, slots collapse, empty slots are frequently false-positive, or derived union outputs disagree strongly with raw union targets.

## Forward Consistency Entry

COMSOL should not be placed inside the training loop. The future path is:

`geometry slots -> derived profile/mask/depth -> lightweight forward surrogate or feature-space residual -> Bx/By/Bz residual`

Forward consistency should first act as evaluator / refinement referee, then only later be considered as a training loss. The existing surface forward-refinement companion remains RBC-representable and does not replace 20.85; multi-pit needs its own geometry-primary companion after slot learning is validated.

## Completion Boundary

The current project closeout does not require multi-pit stable inference. The completion package is clearer if it states that 20.85 remains stable, the liftoff companion is built, the forward-refinement companion is closed, multi-pit dataset/failure evidence is complete, the raster-target mainline is stopped, and the geometry-primary next route is explicit. This is not multi-pit baseline success and does not update `CURRENT_BASELINE.md`.
