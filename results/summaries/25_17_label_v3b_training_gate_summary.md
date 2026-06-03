# 25.17 Label-V3B Component-Set Training Gate

- gate_decision: `PARTIAL`
- dataset_id: `comsol_surface_multipit_component_set_pilot_v1`
- model_route: `C1_fixed_K_component_set_lightweight_gate`
- loss_config: `component_set_gate_v1`
- target_version: `v3b`
- split: `{'train': 72, 'val': 20, 'test': 20}`
- selected_existence_threshold: `0.55`
- best_epoch: `4`
- first_train_loss: `2.612341`
- final_train_loss: `0.337939`
- best_val_loss: `1.976693`
- final_train_mask_depth_weighted_ratio: `0.972504`
- final_val_mask_depth_weighted_ratio: `0.424539`
- final_train_component_separation_weighted_ratio: `0.000000`
- final_val_component_separation_weighted_ratio: `0.000000`
- final_train_component_mask_to_union_mask_weighted_ratio: `327931513388.951599`
- final_train_label_v3_sdf_loss: `0.065148`
- final_val_label_v3_sdf_loss: `0.177909`
- final_train_label_v3_soft_bce_loss: `0.271450`
- final_val_label_v3_soft_bce_loss: `0.842804`
- final_train_label_v3_valid_depth_loss: `0.000795`
- final_val_label_v3_valid_depth_loss: `0.041506`

## Target Transform Usage

- component_mask_target_v3_soft: `False`
- component_sdf_target_v3: `False`
- component_valid_region_mask: `False`
- component_depth_target_v3: `False`
- component_hard_core_mask_v3b: `True`
- component_boundary_halo_mask_v3b: `True`
- component_ignore_overlap_mask_v3b: `True`
- component_mask_target_v3b_soft: `True`
- component_sdf_target_v3b: `True`
- component_valid_region_mask_v3b: `True`
- component_depth_target_v3b: `True`
- component_identity_conflict_mask_v3b: `True`
- v3b_hard_core_pixel_mean/min: `99.851695` / `47`
- v3b_boundary_halo_pixel_mean/min: `24.728814` / `10`
- v3b_soft_support_pixel_mean/min: `124.580508` / `57`
- v3b_soft_or_raw_union_ratio_mean/max: `1.247726` / `1.250000`
- v3b_identity_conflict_pixel_sum: `915`
- v3b_ignore_overlap_pixel_sum: `472`
- v3_target_loaded_count: `None`
- v3_soft_support_pixel_mean/min: `null` / `None`
- v3_valid_region_pixel_mean/min: `null` / `None`
- v3_depth_valid_region_pixel_mean/min: `null` / `None`
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
- extra_rate: `0.357143`
- union_mask_dice: `0.059298`
- depth_grid_RMSE_m: `0.001191190`

## Test Metrics

- component_recall: `0.674419`
- missed_rate: `0.325581`
- merged_rate: `1.000000`
- extra_rate: `0.292683`
- center_error_m_mean: `0.004714954`
- lwd_relative_error_mean: `0.149472`
- rotation_error_rad_mean: `0.541071`
- component_mask_dice: `0.034007`
- union_mask_dice: `0.060961`
- depth_grid_RMSE_m: `0.001178787`

## Required Test Subsets

- component_count=3: recall `0.666667`, merged `1.000000`, component Dice `0.036607`, union Dice `0.082227`
- partially_overlapping: recall `0.666667`, merged `1.000000`, component Dice `0.039036`, union Dice `0.065059`
- touching_boundary: recall `0.444444`, merged `1.000000`, component Dice `0.026335`, union Dice `0.060115`

## 25.10 Comparison

- component_recall_delta: `-0.162791`
- component_mask_dice_delta: `-0.075555`
- union_mask_dice_delta: `-0.069519`
- center_error_delta_m: `0.000430942`
- lwd_relative_error_delta: `-0.005080`
- depth_grid_RMSE_delta_m: `0.000935472`

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
- extra_rate_delta: `0.000000`
- merged_rate_delta: `1.000000`
- component_mask_dice_delta: `0.028471`
- union_mask_dice_delta: `0.058132`
- depth_grid_RMSE_delta_m: `0.000935896`

## 25.15 Comparison

- component_recall_delta: `0.000000`
- missed_rate_delta: `0.000000`
- extra_rate_delta: `-0.048226`
- merged_rate_delta: `0.000000`
- component_mask_dice_delta: `-0.000239`
- union_mask_dice_delta: `-0.000733`
- depth_grid_RMSE_delta_m: `0.000072564`

## Boundary

- This is a training gate, not a baseline replacement.
- No `CURRENT_BASELINE.md` transition is authorized.
- Model architecture, K=3 component-set representation, fixed split, and Hungarian matching are unchanged.
- 25.17 uses the 25.10 loss mainline with explicit label supervision, not the 25.11/25.12 rebalance stack.
- No checkpoint or generated data artifact is committed by this gate.
- next_route: `B. enter 25.17b label-v3b failure audit focused on hard-core/halo/SDF/depth-valid-region usage`
