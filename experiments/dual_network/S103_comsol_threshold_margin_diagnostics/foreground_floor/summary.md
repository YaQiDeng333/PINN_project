# COMSOL threshold margin diagnostics

- run_dir: `experiments\dual_network\S101_comsol_v2_collapse_suppression_probe\foreground_floor`
- hard_area_zero: `True`
- soft_hard_mismatch: `True`
- no_threshold_crossing: `True`
- soft_collapse: `False`
- foreground_collapse: `False`
- final_min_mu: `6.272391e+02`
- final_mean_mu: `6.273063e+02`
- final_mean_soft_defect: `7.268819e-02`
- final_pred_area_soft_mean: `2.977308e+02`
- final_true_area_mean: `2.114400e+02`

## 判断

需要 threshold-margin loss：soft foreground 已非零但 hard mask 仍为 0，且 mu_pred 没有跨过阈值。