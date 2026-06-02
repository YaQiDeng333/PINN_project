# 25.15 Label-V3 Component-Set Training Gate

- gate_decision: `FAIL`
- dataset_id: `comsol_surface_multipit_component_set_pilot_v1`
- model_route: `C1_fixed_K_component_set_lightweight_gate`
- loss_config: `component_set_gate_v1`
- target_version: `v3`
- split: `{'test': 20, 'train': 72, 'val': 20}`
- selected_existence_threshold: `0.25`
- best_epoch: `3`
- first_train_loss: `2.670443`
- final_train_loss: `0.468332`
- best_val_loss: `2.113839`
- final_train_mask_depth_weighted_ratio: `0.980730`
- final_val_mask_depth_weighted_ratio: `0.450453`
- final_train_component_separation_weighted_ratio: `0.000000`
- final_val_component_separation_weighted_ratio: `0.000000`
- final_train_component_mask_to_union_mask_weighted_ratio: `458561281363.169373`
- final_train_label_v3_sdf_loss: `0.055944`
- final_val_label_v3_sdf_loss: `0.255177`
- final_train_label_v3_soft_bce_loss: `0.362142`
- final_val_label_v3_soft_bce_loss: `0.896596`
- final_train_label_v3_valid_depth_loss: `0.000829`
- final_val_label_v3_valid_depth_loss: `0.044922`

## Target Transform Usage

- component_mask_target_v3_soft: `True`
- component_sdf_target_v3: `True`
- component_valid_region_mask: `True`
- component_depth_target_v3: `True`
- v3_target_loaded_count: `112`
- v3_soft_support_pixel_mean/min: `210.110169` / `126`
- v3_valid_region_pixel_mean/min: `210.110169` / `126`
- v3_depth_valid_region_pixel_mean/min: `101.110169` / `58`
- empty_slot_violation_count: `0`
- component_mask_target_v2: `False`
- component_depth_target_v2: `False`
- component_ownership_map: `True`
- target_loaded_count: `112`
- ownership_resolved_pixel_count: `23565`
- ownership_resolved_overlap_pixel_count: `290`
- duplicate_ownership_before_v2: `297`
- duplicate_ownership_after_v2: `0`
- overlap_depth_conflict_before_v2: `271`
- overlap_depth_conflict_after_v2: `0`

## Validation Metrics

- component_recall: `0.642857`
- missed_rate: `0.357143`
- merged_rate: `1.000000`
- extra_rate: `0.386364`
- union_mask_dice: `0.058491`
- depth_grid_RMSE_m: `0.001108671`

## Test Metrics

- component_recall: `0.674419`
- missed_rate: `0.325581`
- merged_rate: `1.000000`
- extra_rate: `0.340909`
- center_error_m_mean: `0.004201829`
- lwd_relative_error_mean: `0.170478`
- rotation_error_rad_mean: `0.598689`
- component_mask_dice: `0.034245`
- union_mask_dice: `0.061694`
- depth_grid_RMSE_m: `0.001106223`

## Required Test Subsets

- component_count=3: recall `0.777778`, merged `1.000000`, component Dice `0.037991`, union Dice `0.082585`
- partially_overlapping: recall `0.666667`, merged `1.000000`, component Dice `0.036652`, union Dice `0.066119`
- touching_boundary: recall `0.555556`, merged `1.000000`, component Dice `0.027735`, union Dice `0.061031`

## 25.10 Comparison

- component_recall_delta: `-0.162791`
- component_mask_dice_delta: `-0.075316`
- union_mask_dice_delta: `-0.068785`
- center_error_delta_m: `-0.000082183`
- lwd_relative_error_delta: `0.015925`
- depth_grid_RMSE_delta_m: `0.000862908`

## 25.11 Comparison

- component_recall_delta: `null`
- missed_rate_delta: `null`
- extra_rate_delta: `null`
- merged_rate_delta: `null`
- component_mask_dice_delta: `null`
- union_mask_dice_delta: `null`
- depth_grid_RMSE_delta_m: `null`

## 25.12 Comparison

- component_recall_delta: `null`
- missed_rate_delta: `null`
- extra_rate_delta: `null`
- merged_rate_delta: `null`
- component_mask_dice_delta: `null`
- union_mask_dice_delta: `null`
- depth_grid_RMSE_delta_m: `null`

## 25.13 Comparison

- component_recall_delta: `0.000000`
- missed_rate_delta: `0.000000`
- extra_rate_delta: `0.048226`
- merged_rate_delta: `1.000000`
- component_mask_dice_delta: `0.028710`
- union_mask_dice_delta: `0.058865`
- depth_grid_RMSE_delta_m: `0.000863332`

## Boundary

- This is a training gate, not a baseline replacement.
- No `CURRENT_BASELINE.md` transition is authorized.
- Model architecture, K=3 component-set representation, fixed split, and Hungarian matching are unchanged.
- 25.15 uses the 25.10 loss mainline with label-v3 loader targets, not the 25.11/25.12 rebalance stack.
- No checkpoint or generated data artifact is committed by this gate.
- next_route: `C. return to label-v3 derivation or generator/export schema; do not continue loss tuning`
