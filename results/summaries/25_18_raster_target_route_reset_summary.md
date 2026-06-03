# 25.18 Raster-Target Route Reset

- route reset decision: `STOP_RASTER_TARGET_MAINLINE`
- next route: `A. enter 25.19 geometry-primary component-set design + label derivation plan; no training`
- evidence sufficient: `True`
- missing evidence files: `0`

## Main Conclusion

Stop the per-component raster-target mainline and move to geometry-primary component-set design.

The failure pattern is route-level, not a single-run bug: target-v2 becomes near-empty after ownership cleanup, label-v3 becomes union-like after soft support, and label-v3b still has merged_rate `1.000000` after hard-core/halo/SDF supervision.

## Evidence Snapshot

| stage | decision | recall | missed | extra | merged | component Dice | union Dice | depth RMSE m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25.10 | PARTIAL | 0.837209 | 0.162791 | 0.142857 | 0.200000 | 0.109562 | 0.130480 | 0.000243315 |
| 25.13 | FAIL | 0.674419 | 0.325581 | 0.292683 | 0.000000 | 0.005536 | 0.002829 | 0.000242891 |
| 25.15 | FAIL | 0.674419 | 0.325581 | 0.340909 | 1.000000 | 0.034245 | 0.061694 | 0.001106223 |
| 25.17 | PARTIAL | 0.674419 | 0.325581 | 0.292683 | 1.000000 | 0.034007 | 0.060961 | 0.001178787 |

## Stop Routes

- `label-v2 target-v2 training route`: `STOP`. 25.13 target-v2 training gate FAIL: ownership cleanup removed duplicate/overlap conflict but produced near-empty mask collapse.
- `label-v3 soft support training route`: `STOP`. 25.15 label-v3 training gate FAIL: soft support relieved sparsity but produced union-like merged collapse.
- `label-v3b hard-core/halo/SDF raster-supervision route`: `STOP`. 25.17 label-v3b training gate PARTIAL: near-empty was partly relieved, but merged_rate remained 1.000000.
- `loss rebalance / label-v4 raster-target tuning as the mainline`: `STOP`. 25.11/25.12 rebalance attempts either caused merge collapse or failed; the route now points away from raster-target main supervision.

## Preserve Routes

- multi-pit component-set direction
- K=3 slot representation
- component geometry prediction
- future forward consistency
- raw labels and COMSOL top-up dataset as evidence/source data

## Geometry-Primary Route

- slot primary outputs: `existence_prob`, `center_x_m`, `center_y_m`, `L_m`, `W_m`, `D_m`, `rotation_angle`, `shape_family`, `compact_shape_parameters`
- derived outputs: `derived_component_mask`, `derived_union_mask`, `derived_component_depth`, `derived_union_depth`
- supervision boundary: per-component raster targets may remain auxiliary diagnostics or weak supervision, but are no longer the main loss.
- matching priority: `existence, center, L/W/D, rotation, optional derived union consistency`
- forward consistency: geometry slots -> derived profile/mask/depth -> lightweight forward surrogate or feature-space residual -> Bx/By/Bz residual; COMSOL is not placed inside the training loop.

## Roadmap

- `25.19`: geometry-primary component-set design + label derivation plan; training=`False`
- `25.20`: two-component separated/close geometry-primary training gate; training=`True`
- `25.21`: geometry-derived mask/depth evaluator + forward-consistency surrogate plan; training=`False`
- `25.22`: topology-aware expansion for touching/overlap/three-component; training=`gate-dependent`

## Boundary

- training_run: `False`
- loss_tuning: `False`
- model_capacity_expanded: `False`
- comsol_run: `False`
- data_npz_modified: `False`
- current_baseline_updated: `False`
- baseline_transition: `False`
- continues_label_v4_or_loss_v5_raster_training: `False`
