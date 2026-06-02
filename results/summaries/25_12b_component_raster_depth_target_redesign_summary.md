# 25.12b Component Raster/Depth Target Redesign

- target_redesign_acceptance_decision: `READY_FOR_25.13_TRAINING`
- route_decision: `A. enter 25.13 target-v2 training gate using the 25.10 loss mainline; do not use the 25.11/25.12 rebalance stack`
- main_conclusion: v1 generator labels are globally consistent, but component-level raster/depth supervision lacks explicit ownership in overlap/touching pixels; target v2 should resolve ownership before further training and return to the 25.10 loss mainline.

## Evidence From 25.10 -> 25.12

- 25.10: recall `0.837209`, merged `0.200000`, component Dice `0.109562`, union Dice `0.130480`, depth RMSE `0.000243315 m`.
- 25.11: recall `0.860465`, merged `0.900000`, component Dice `0.108737`, union Dice `0.166233`, depth RMSE `0.000673627 m`.
- 25.12: recall `0.744186`, merged `0.700000`, component Dice `0.108790`, union Dice `0.138075`, depth RMSE `0.000501023 m`.

## V1 Target Audit

- component OR to union Dice mean/min: `1.000000` / `1.000000`.
- max(component depth) to union depth RMSE mean/max: `0.000000000` / `0.000000000 m`.
- empty-slot mask/depth violations: `0`.
- center-to-mask-centroid error mean/p95/max: `0.000059604` / `0.000108224` / `0.001045136 m`.
- overlap samples: `25/112`; duplicated component target pixels: `297`.
- partially_overlapping overlap samples: `18/24`.
- touching overlap samples: `4/24`.
- component_count=3 overlap samples: `10/12`.

## V1 Main Problems

- Component-level masks are not ownership-resolved in overlap pixels, so one raster pixel can be a positive target for multiple slots.
- Union mask/depth targets hide that ambiguity because OR/max exactly reconstructs the sample-level target.
- Loss-only rebalancing can therefore improve or preserve union-level agreement while failing component-level separation.
- Three-component rows concentrate overlap ambiguity and still require a mandatory separate slice.

## V2 Core Rules

- `component_mask_target_v2`: per-slot binary masks are ownership-resolved; each foreground pixel belongs to at most one component.
- `component_depth_target_v2`: component depth is foreground-only and ownership-resolved; background does not dominate depth loss.
- `component_ownership_map`: `-1` for background and `0..K-1` for owning slot, with nearest normalized center, deeper local depth, then slot id as deterministic tie-breaks.
- `overlap_policy`: separated/close rows must be mutually exclusive; touching rows may share continuous boundaries but not raster ownership; partially-overlapping rows keep raw overlap diagnostics while training on ownership-resolved component targets.
- `union_from_components_rule`: union mask is OR and union depth is max over raw components, never sum.
- 25.13 should use the 25.10 loss mainline plus target-v2 transform, not the 25.11/25.12 rebalance stack.

## Boundary

- This stage did not train a model.
- It did not run COMSOL.
- It did not modify data/NPZ files.
- It did not modify `CURRENT_BASELINE.md` or authorize a baseline transition.
