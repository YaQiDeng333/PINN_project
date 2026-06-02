# 25.11 Mask/Depth Loss Rebalance Training

- gate_decision: `PARTIAL`
- dataset_id: `comsol_surface_multipit_component_set_pilot_v1`
- model_route: `C1_fixed_K_component_set_lightweight_gate`
- loss_config: `mask_depth_rebalance_v1`
- split: `{'train': 72, 'val': 20, 'test': 20}`
- selected_existence_threshold: `0.35`
- best_epoch: `3`
- first_train_loss: `6.458076`
- final_train_loss: `0.454230`
- best_val_loss: `5.536524`
- final_train_mask_depth_weighted_ratio: `0.973279`
- final_val_mask_depth_weighted_ratio: `0.929449`

## Validation Metrics

- component_recall: `0.928571`
- missed_rate: `0.071429`
- merged_rate: `0.900000`
- extra_rate: `0.048780`
- union_mask_dice: `0.161905`
- depth_grid_RMSE_m: `0.000648111`

## Test Metrics

- component_recall: `0.860465`
- missed_rate: `0.139535`
- merged_rate: `0.900000`
- extra_rate: `0.097561`
- center_error_m_mean: `0.005085919`
- lwd_relative_error_mean: `0.155758`
- rotation_error_rad_mean: `0.582618`
- component_mask_dice: `0.108737`
- union_mask_dice: `0.166233`
- depth_grid_RMSE_m: `0.000673627`

## 25.10 Comparison

- component_recall_delta: `0.023256`
- component_mask_dice_delta: `-0.000825`
- union_mask_dice_delta: `0.035753`
- center_error_delta_m: `0.000801907`
- lwd_relative_error_delta: `0.001206`
- depth_grid_RMSE_delta_m: `0.000430312`

## Boundary

- This is a training gate, not a baseline replacement.
- No `CURRENT_BASELINE.md` transition is authorized.
- Model architecture, K=3 component-set representation, fixed split, and Hungarian matching are unchanged.
- No checkpoint or generated data artifact is committed by this gate.
- next_route: `B. run 25.11b targeted rebalance or topology-focused failure audit`
