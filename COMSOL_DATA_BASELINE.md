# COMSOL_DATA_BASELINE

## Baseline

- Name: COMSOL single-defect pilot_v9 mask-only multi-line baseline.
- Dataset: `comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect`.
- NPZ path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz`.
- Input: `delta_bz`, shape `(N, 3, 201)`.
- Output: 2D / quasi-2D defect mask, shape `(64, 128)`.
- Model family: mask-only grid decoder with Conv1d Bz encoder and ConvTranspose2d mask decoder.
- Seeds: 42, 123, 2026.
- Threshold selection: validation-only global threshold per seed from candidates `0.30..0.90`.
- Checkpoint selection: validation `IoU + Dice - area_error`, scanning threshold candidates each epoch.

## Results

- Train mean +/- std: iou_mean=0.7093+/-0.0183, dice_mean=0.8272+/-0.0127, area_error_mean=0.1503+/-0.0155, center_error_mean=2.2339+/-0.1090, pred_area_zero_sum=0.0000+/-0.0000
- Val mean +/- std: iou_mean=0.6700+/-0.0030, dice_mean=0.7994+/-0.0023, area_error_mean=0.2009+/-0.0110, center_error_mean=2.5645+/-0.0549, pred_area_zero_sum=0.0000+/-0.0000
- Test mean +/- std: iou_mean=0.6515+/-0.0064, dice_mean=0.7861+/-0.0046, area_error_mean=0.2208+/-0.0099, center_error_mean=2.5623+/-0.0538, pred_area_zero_sum=0.0000+/-0.0000
- Per-seed best epoch / threshold: {42: {'best_epoch': 19, 'checkpoint_threshold': 0.5, 'selected_threshold': 0.5, 'best_val_score': 1.2562607966169401}, 123: {'best_epoch': 23, 'checkpoint_threshold': 0.9, 'selected_threshold': 0.9, 'best_val_score': 1.287768572197726}, 2026: {'best_epoch': 32, 'checkpoint_threshold': 0.8, 'selected_threshold': 0.8, 'best_val_score': 1.261351666295834}}

## Group Checks

- Defect type summary: `C:\Users\19166\Desktop\PINN_project\results\metrics\comsol_pilot_v9_baseline_defect_type_summary.csv`.
- Angle summary: `C:\Users\19166\Desktop\PINN_project\results\metrics\comsol_pilot_v9_baseline_angle_summary.csv`.
- Vertex count summary: `C:\Users\19166\Desktop\PINN_project\results\metrics\comsol_pilot_v9_baseline_vertex_count_summary.csv`.
- Source pack summary: `C:\Users\19166\Desktop\PINN_project\results\metrics\comsol_pilot_v9_baseline_source_pack_summary.csv`.

## Scope And Limitations

- This is a COMSOL data-domain baseline only.
- It does not replace `CURRENT_BASELINE.md`.
- It is not a `v3_complex` baseline and should not be compared as a formal replacement.
- It is pilot-level controlled synthetic data.
- It covers single-defect rectangular_notch, rotated_rect, and polygon only.
- It does not demonstrate multi_defect capability or real experimental data generalization.
