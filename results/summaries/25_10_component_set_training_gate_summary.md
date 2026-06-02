# 25.10 Surface Multi-Pit Component-Set Training Gate

- gate_decision: `PARTIAL`
- dataset_id: `comsol_surface_multipit_component_set_pilot_v1`
- model_route: `C1_fixed_K_component_set_lightweight_gate`
- split: `{'train': 72, 'val': 20, 'test': 20}`
- selected_existence_threshold: `0.25`
- best_epoch: `6`
- first_train_loss: `3.014077`
- final_train_loss: `0.090511`
- best_val_loss: `1.992010`

## Validation Metrics

- component_recall: `0.785714`
- missed_rate: `0.214286`
- merged_rate: `0.000000`
- extra_rate: `0.232558`
- union_mask_dice: `0.071430`
- depth_grid_RMSE_m: `0.000206167`

## Test Metrics

- component_recall: `0.837209`
- missed_rate: `0.162791`
- merged_rate: `0.200000`
- extra_rate: `0.142857`
- center_error_m_mean: `0.004284012`
- lwd_relative_error_mean: `0.154552`
- rotation_error_rad_mean: `0.575395`
- component_mask_dice: `0.109562`
- union_mask_dice: `0.130480`
- depth_grid_RMSE_m: `0.000243315`

## Boundary

- This is a training gate, not a baseline replacement.
- No `CURRENT_BASELINE.md` transition is authorized.
- No checkpoint or generated data artifact is committed by this gate.
- next_route: `B. run 25.10b failure audit for merged/missed, overlap/touching, slot permutation, and three-component rows`
