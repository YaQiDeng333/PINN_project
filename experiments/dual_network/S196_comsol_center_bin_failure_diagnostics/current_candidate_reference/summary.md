# Center-bin failure diagnostics: current_candidate_reference

- prediction_dir: `experiments\dual_network\S191_comsol_signal_to_center_aux_quick_gate\current_candidate_reference`
- diagnosed_splits: `train, val, test`
- missing_splits: `none`
- val_mean_iou: `0.546311`
- test_mean_iou: `0.586546`
- val_x_wrong_rate: `0.200000`
- val_y_wrong_rate: `0.083333`
- test_x_wrong_rate: `0.133333`
- test_y_wrong_rate: `0.033333`
- val_mean_center_grid_error: `3.250845`
- test_mean_center_grid_error: `2.572883`
- worst_val_sample_count_below_050: `7`

## Answers

1. x-bin vs y-bin: `x-bin` has the higher val wrong-rate in this run.
2. Val fluctuation should be checked through `worst_samples.csv`; the count below IoU 0.50 is listed above.
3. The most error-prone component slot can be read from `grouped_center_bin_errors.csv` where `group_by=component_slot`.
4. Type / rotation / area bin sensitivity is reported in grouped rows for `type_true`, `rotation_bin`, and `target_area_bin`.
5. Auxiliary-head usefulness is decided by comparing this summary across labels, not within a single run.
6. Preferred next action depends on S196 aggregate comparison; do not infer a new model route from this single-run summary alone.
