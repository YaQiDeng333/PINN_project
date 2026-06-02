# 25.13 Target-V2 Component-Set Training Gate

- gate_decision: `FAIL`
- dataset_id: `comsol_surface_multipit_component_set_pilot_v1`
- model_route: `C1_fixed_K_component_set_lightweight_gate`
- loss_config: `component_set_gate_v1`
- target_version: `v2`
- split: `{'train': 72, 'val': 20, 'test': 20}`
- selected_existence_threshold: `0.55`
- best_epoch: `4`
- first_train_loss: `3.014221`
- final_train_loss: `0.090364`
- best_val_loss: `1.973384`
- final_train_mask_depth_weighted_ratio: `0.865098`
- final_val_mask_depth_weighted_ratio: `0.357858`
- final_train_component_separation_weighted_ratio: `0.000000`
- final_val_component_separation_weighted_ratio: `0.000000`
- final_train_component_mask_to_union_mask_weighted_ratio: `78082788735.628128`

## Target V2 Usage

- component_mask_target_v2: `True`
- component_depth_target_v2: `True`
- component_ownership_map: `True`
- target_loaded_count: `112`
- ownership_resolved_pixel_count: `23565`
- ownership_resolved_overlap_pixel_count: `290`
- duplicate_ownership_before_v2: `297`
- duplicate_ownership_after_v2: `0`
- overlap_depth_conflict_before_v2: `271`
- overlap_depth_conflict_after_v2: `0`

## Validation Metrics

- component_recall: `0.738095`
- missed_rate: `0.261905`
- merged_rate: `0.000000`
- extra_rate: `0.261905`
- union_mask_dice: `0.003187`
- depth_grid_RMSE_m: `0.000204975`

## Test Metrics

- component_recall: `0.674419`
- missed_rate: `0.325581`
- merged_rate: `0.000000`
- extra_rate: `0.292683`
- center_error_m_mean: `0.004344451`
- lwd_relative_error_mean: `0.167281`
- rotation_error_rad_mean: `0.532413`
- component_mask_dice: `0.005536`
- union_mask_dice: `0.002829`
- depth_grid_RMSE_m: `0.000242891`

## 25.10 Comparison

- component_recall_delta: `-0.162791`
- component_mask_dice_delta: `-0.104026`
- union_mask_dice_delta: `-0.127651`
- center_error_delta_m: `0.000060439`
- lwd_relative_error_delta: `0.012729`
- depth_grid_RMSE_delta_m: `-0.000000424`

## 25.11 Comparison

- component_recall_delta: `-0.186047`
- missed_rate_delta: `0.186047`
- extra_rate_delta: `0.195122`
- merged_rate_delta: `-0.900000`
- component_mask_dice_delta: `-0.103201`
- union_mask_dice_delta: `-0.163403`
- depth_grid_RMSE_delta_m: `-0.000430736`

## 25.12 Comparison

- component_recall_delta: `-0.069767`
- missed_rate_delta: `0.069767`
- extra_rate_delta: `0.092683`
- merged_rate_delta: `-0.700000`
- component_mask_dice_delta: `-0.103255`
- union_mask_dice_delta: `-0.135246`
- depth_grid_RMSE_delta_m: `-0.000258132`

## Boundary

- This is a training gate, not a baseline replacement.
- No `CURRENT_BASELINE.md` transition is authorized.
- Model architecture, K=3 component-set representation, fixed split, and Hungarian matching are unchanged.
- 25.13 uses the 25.10 loss mainline with target-v2 loader targets, not the 25.11/25.12 rebalance stack.
- No checkpoint or generated data artifact is committed by this gate.
- next_route: `C. return to generator/label schema; do not continue loss tuning`
