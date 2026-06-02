# 25.13b Generator/Label Schema Audit After Target-V2 Collapse

- acceptance_decision: `NEEDS_PINN_LABEL_DERIVATION_V3`
- route_decision: `A. enter 25.14 label-v3 derivation + validator, no training`
- main_conclusion: 25.13 target-v2 collapse is not caused by empty v2 components or broad generator corruption; the current hard binary component-local labels are too sparse/unsupported for stable mask learning, so derive label schema v3 soft/valid-region targets in PINN_project before any further training.

## 25.13 Collapse Evidence

- recall: `0.674419`
- missed: `0.325581`
- extra: `0.292683`
- merged: `0.000000`
- component Dice: `0.005536`
- union Dice: `0.002829`
- depth RMSE m: `0.000242891`

## Target-V2 Support Audit

- duplicate ownership: `297 -> 0`
- overlap-depth-conflict: `271 -> 0`
- active components: `236`
- v2 foreground px mean/min/p05: `99.851695` / `47` / `58.750000`
- v2 positive fraction mean: `0.012188928`
- v2/v1 shrink ratio mean/min: `0.985320` / `0.746032`
- empty existing v2 masks: `0`
- tiny existing v2 masks <20 px: `0`
- shrink ratio <0.80: `3`
- partially_overlapping shrink ratio mean/min: `0.961937` / `0.746032`
- component_count=3 shrink ratio mean/min: `0.923066` / `0.746032`

## Diagnosis

- V2 ownership resolution is not deleting whole components: existing slots stay non-empty and average support remains near 100 pixels.
- The support is still very sparse on a 64x128 grid, so hard binary component targets have weak positive signal and no boundary/context target.
- The zero merged rate in 25.13 is a near-empty mask artifact, not successful component separation.
- Full-grid depth RMSE staying stable does not prove component-depth learning because mask collapse reduces effective component evidence.

## Label Schema V3 Recommendation

- Preserve `raw_component_mask_raw` and `component_ownership_map`.
- Add `component_mask_target_v3_soft` or `component_sdf_target_v3`.
- Add `component_valid_region_mask`, `overlap_region_mask`, and `contact_boundary_mask`.
- Add `component_depth_target_v3` with an explicit valid region.
- Keep union mask/depth as OR/max from raw components for evaluation comparability.
- Derive v3 labels inside `PINN_project` first; no COMSOL generator change is required yet.

## Boundary

- This audit did not train a model or tune losses.
- It did not run COMSOL or modify data/NPZ files.
- It did not modify `CURRENT_BASELINE.md` or authorize a baseline transition.
