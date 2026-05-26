# COMSOL threshold margin diagnostics

- run_dir: `experiments\dual_network\S101_comsol_v2_collapse_suppression_probe\v2_baseline_with_history`
- hard_area_zero: `True`
- soft_hard_mismatch: `True`
- no_threshold_crossing: `True`
- soft_collapse: `False`
- foreground_collapse: `False`
- final_min_mu: `6.258309e+02`
- final_mean_mu: `6.259980e+02`
- final_mean_soft_defect: `7.447070e-02`
- final_pred_area_soft_mean: `3.050320e+02`
- final_true_area_mean: `2.187300e+02`

## 判断

需要 threshold-margin loss：soft foreground 已非零但 hard mask 仍为 0，且 mu_pred 没有跨过阈值。