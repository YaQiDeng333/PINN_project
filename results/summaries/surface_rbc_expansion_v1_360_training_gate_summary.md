# Surface RBC Expansion v1 360 Training Gate

- assembled_dataset_id: `comsol_true_3d_rbc_surface_expansion_v1_360`
- N: 360
- split: {'train': 242, 'val': 59, 'test': 59}
- selected_seed: 2026
- validation_pass: True
- gate_outcome: FAIL
- gate_reason: old_v3_240_test non-regression threshold failed
- baseline_ready: false
- CURRENT_BASELINE_update: false

## Old v3_240 Test Non-Regression

- profile_depth_rmse_m: 0.000429130307356 <= 0.00039936911
- Er-like: 0.432564256474 <= 0.3575712
- L_MAE_mm: 1.93214609264 <= 1.9866
- W_MAE_mm: 1.89015141473 <= 2.2953
- D_MAE_mm: 0.9211626135 <= 0.84
- projected_mask_Dice: 0.860724439354 >= 0.837727

## Comparator Summary

- topup_test primary improvements vs 20.85: 4/4
- assembled_test primary improvements vs 20.85: 1/4
- hard-bin improved sets: 9/22
- topup_test profile RMSE candidate/baseline: 0.000527715086355 / 0.00059367757658
- topup_test Er-like candidate/baseline: 0.2789888056 / 0.279104615003
- topup_test D MAE mm candidate/baseline: 0.913140838966 / 1.11393855512
- topup_test Dice candidate/baseline: 0.873616481995 / 0.851776822476
- assembled_test profile RMSE candidate/baseline: 0.000462548876508 / 0.000457547539246
- assembled_test Er-like candidate/baseline: 0.380504781601 / 0.319716887959
- assembled_test D MAE mm candidate/baseline: 0.918443367895 / 0.906572773895
- assembled_test Dice candidate/baseline: 0.8650946233 / 0.849099911524

Boundary: this gate creates an assembled candidate dataset for explicit review only. It is not a baseline transition.
