# COMSOL threshold margin diagnostics

- run_dir: `experiments\dual_network\S101_comsol_v2_collapse_suppression_probe\area_ratio_mse`
- hard_area_zero: `True`
- soft_hard_mismatch: `True`
- no_threshold_crossing: `True`
- soft_collapse: `False`
- foreground_collapse: `False`
- final_min_mu: `5.413740e+02`
- final_mean_mu: `6.736925e+02`
- final_mean_soft_defect: `8.250425e-02`
- final_pred_area_soft_mean: `3.379374e+02`
- final_true_area_mean: `2.264800e+02`

## 判断

需要 threshold-margin loss：soft foreground 已非零但 hard mask 仍为 0，且 mu_pred 没有跨过阈值。