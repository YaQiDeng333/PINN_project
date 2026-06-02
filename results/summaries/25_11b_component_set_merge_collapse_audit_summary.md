# 25.11b Component-Set Merge-Collapse Audit

- audit_conclusion: 25.11 primarily induced union-over-component merge collapse: union Dice improved while component separation and depth consistency degraded.
- top_failure: `union-over-component collapse`
- next_route: `A. enter 25.12 component-separation-aware rebalance training`

## 25.10 -> 25.11 Test Delta

- component_recall: `0.837209 -> 0.860465`
- merged_rate: `0.200000 -> 0.900000`
- component_mask_dice: `0.109562 -> 0.108737`
- union_mask_dice: `0.130480 -> 0.166233`
- depth_grid_RMSE_m: `0.000243315 -> 0.000673627`

## Collapse Evidence

- test_newly_merged_rate: `0.700000`
- test_union_over_component_collapse_rate: `0.500000`
- separated_newly_merged_rate: `0.750000`
- depth_supervision_dilution_rate: `0.600000`
- final_val_mask_depth_weighted_ratio: `0.929449`

## Boundary

- This audit did not train a new model.
- It did not run COMSOL or modify data/NPZ files.
- It did not modify `CURRENT_BASELINE.md`.
- It does not authorize a baseline transition.
