# s119_refined parametric error diagnostics

- run_dir: `experiments\dual_network\S119_comsol_parametric_inverse_refined_probe\refined_mlp`
- oracle_dir: `experiments\dual_network\S117_comsol_parametric_raster_oracle`

## Split summary

- train: mask_iou=6.689502e-01, oracle_iou=7.229967e-01, oracle_gap=5.404654e-02, type_acc=1.000000e+00, rotation_mae=2.569852e-01, dominant=no_single_dominant_error
- val: mask_iou=3.257646e-01, oracle_iou=7.232882e-01, oracle_gap=3.975235e-01, type_acc=6.166667e-01, rotation_mae=7.278932e+00, dominant=type, rotation, oracle_gap
- test: mask_iou=3.885092e-01, oracle_iou=7.165838e-01, oracle_gap=3.280746e-01, type_acc=6.833333e-01, rotation_mae=7.859528e+00, dominant=type, rotation, oracle_gap

## 当前判断

- val/test avg oracle gap: `3.627991e-01`。
- val/test avg type accuracy: `6.500000e-01`。
- val/test avg rotation MAE: `7.569230e+00` degree。
- 当前 run 目录没有 per-sample predictions，因此本诊断只做 aggregate decomposition；未伪造 type/rotation bins。
- 如果 oracle gap 大且 type/rotation 同时偏弱，下一步优先改 head / encoder / loss 分解，而不是修 target/mask schema。
