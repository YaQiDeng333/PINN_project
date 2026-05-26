# Center-bin failure diagnostics: x_bin_slot_weighted

- prediction_dir: `experiments\dual_network\S200_comsol_x_bin_center_calibration_quick_gate\x_bin_slot_weighted`
- diagnosed_splits: `train, val, test`
- missing_splits: `none`
- val_mean_iou: `0.518272`
- test_mean_iou: `0.543191`
- val_x_wrong_rate: `0.250000`
- val_y_wrong_rate: `0.216667`
- test_x_wrong_rate: `0.183333`
- test_y_wrong_rate: `0.083333`
- val_mean_center_grid_error: `5.811700`
- test_mean_center_grid_error: `3.065946`
- worst_val_sample_count_below_050: `7`

## Answers

1. x-bin vs y-bin: `x-bin` has the higher val wrong-rate in this run.
2. Val fluctuation should be checked through `worst_samples.csv`; the count below IoU 0.50 is listed above.
3. The most error-prone component slot can be read from `grouped_center_bin_errors.csv` where `group_by=component_slot`.
4. Type / rotation / area bin sensitivity is reported in grouped rows for `type_true`, `rotation_bin`, and `target_area_bin`.
5. Auxiliary-head usefulness is decided by comparing this summary across labels, not within a single run.
6. Preferred next action depends on S196 aggregate comparison; do not infer a new model route from this single-run summary alone.
