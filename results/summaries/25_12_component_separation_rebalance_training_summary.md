# 25.12 Component-Separation-Aware Rebalance Training

- gate_decision: `FAIL`
- dataset_id: `comsol_surface_multipit_component_set_pilot_v1`
- model_route: `C1_fixed_K_component_set_lightweight_gate`
- loss_config: `component_separation_rebalance_v1`
- split: `{'train': 72, 'val': 20, 'test': 20}`
- selected_existence_threshold: `0.25`
- best_epoch: `4`
- first_train_loss: `5.967029`
- final_train_loss: `0.301080`
- best_val_loss: `4.359128`
- final_train_mask_depth_weighted_ratio: `0.844802`
- final_val_mask_depth_weighted_ratio: `0.817049`
- final_train_component_separation_weighted_ratio: `0.091104`
- final_val_component_separation_weighted_ratio: `0.035257`
- final_train_component_mask_to_union_mask_weighted_ratio: `11.584615`

## Validation Metrics

- component_recall: `0.785714`
- missed_rate: `0.214286`
- merged_rate: `0.600000`
- extra_rate: `0.175000`
- union_mask_dice: `0.102609`
- depth_grid_RMSE_m: `0.000474734`

## Test Metrics

- component_recall: `0.744186`
- missed_rate: `0.255814`
- merged_rate: `0.700000`
- extra_rate: `0.200000`
- center_error_m_mean: `0.004332302`
- lwd_relative_error_mean: `0.153492`
- rotation_error_rad_mean: `0.598953`
- component_mask_dice: `0.108790`
- union_mask_dice: `0.138075`
- depth_grid_RMSE_m: `0.000501023`

## 25.10 Comparison

- component_recall_delta: `-0.093023`
- component_mask_dice_delta: `-0.000772`
- union_mask_dice_delta: `0.007596`
- center_error_delta_m: `0.000048290`
- lwd_relative_error_delta: `-0.001060`
- depth_grid_RMSE_delta_m: `0.000257708`

## 25.11 Comparison

- component_recall_delta: `-0.116279`
- missed_rate_delta: `0.116279`
- extra_rate_delta: `0.102439`
- merged_rate_delta: `-0.200000`
- component_mask_dice_delta: `0.000053`
- union_mask_dice_delta: `-0.028157`
- depth_grid_RMSE_delta_m: `-0.000172604`

## Boundary

- This is a training gate, not a baseline replacement.
- No `CURRENT_BASELINE.md` transition is authorized.
- Model architecture, K=3 component-set representation, fixed split, and Hungarian matching are unchanged.
- No checkpoint or generated data artifact is committed by this gate.
- next_route: `C. rollback to 25.10 loss mainline and redesign component raster/depth targets before further training`
