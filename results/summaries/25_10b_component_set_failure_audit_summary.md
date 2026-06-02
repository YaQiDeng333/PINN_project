# 25.10b Component-Set Failure Audit

- audit_conclusion: Primary failure is loss imbalance: component existence and coarse geometry learned, but raster/depth supervision did not translate into aligned masks.
- next_route: `B. enter 25.11 mask/depth loss rebalance training`
- primary_failure: `loss imbalance`

## Test Failure Snapshot

- component_recall: `0.837209`
- missed_rate: `0.162791`
- merged_rate: `0.200000`
- extra_rate: `0.142857`
- component_mask_dice: `0.109562`
- union_mask_dice: `0.130480`
- depth_grid_RMSE_m: `0.000243315`

## Three-Component Finding

- split_counts: `{'train': 7, 'val': 2, 'test': 3}`
- test_pred_component_count_counts: `{2: 3}`
- test_merged_rate: `1.0`

## Raster Target Integrity

- target_union_mask_iou_mean: `1.000000`
- center_to_mask_centroid_error_mean_m: `0.000059604`
- empty_slot_mask_sum: `0.0`
- coordinate_bug_likely: `False`

## Boundary

- This audit did not train a new model.
- It did not modify `CURRENT_BASELINE.md`.
- It does not authorize a baseline transition.
- It does not recommend simply increasing model size without addressing the audited failure mode.
