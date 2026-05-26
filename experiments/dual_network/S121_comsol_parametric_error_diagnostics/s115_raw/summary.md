# s115_raw parametric error diagnostics

- run_dir: `experiments\dual_network\S115_comsol_parametric_inverse_training_probe\v2_parametric_inverse`
- oracle_dir: `experiments\dual_network\S117_comsol_parametric_raster_oracle`

## Split summary

- train: mask_iou=6.980716e-01, oracle_iou=7.229967e-01, oracle_gap=2.492512e-02, type_acc=1.000000e+00, rotation_mae=1.854690e-01, dominant=no_single_dominant_error
- val: mask_iou=3.699078e-01, oracle_iou=7.232882e-01, oracle_gap=3.533803e-01, type_acc=6.500000e-01, rotation_mae=7.731843e+00, dominant=type, rotation, oracle_gap
- test: mask_iou=4.244624e-01, oracle_iou=7.165838e-01, oracle_gap=2.921214e-01, type_acc=6.666667e-01, rotation_mae=7.740396e+00, dominant=type, rotation, oracle_gap

## 当前判断

- val/test avg oracle gap: `3.227509e-01`。
- val/test avg type accuracy: `6.583333e-01`。
- val/test avg rotation MAE: `7.736119e+00` degree。
- 当前 run 目录没有 per-sample predictions，因此本诊断只做 aggregate decomposition；未伪造 type/rotation bins。
- 如果 oracle gap 大且 type/rotation 同时偏弱，下一步优先改 head / encoder / loss 分解，而不是修 target/mask schema。
