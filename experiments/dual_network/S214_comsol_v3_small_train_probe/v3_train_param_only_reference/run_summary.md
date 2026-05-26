# COMSOL parametric inverse run summary

- train_npz: `experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted/train_comsol_v3_hard_case.npz`
- val_npz: `experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted/val_comsol_v3_hard_case.npz`
- test_npz: `experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted/test_comsol_v3_hard_case.npz`
- train_targets: `experiments/dual_network/S209_comsol_v3_hard_case_parametric_targets/train/parametric_targets.npz`
- seed: `1`
- steps: `1500`
- lr: `0.001`
- hidden_dim: `128`
- latent_dim: `64`
- max_components: `3`
- encoder_type: `mlp`
- head_mode: `shared`
- feature_fusion_mode: `none`
- feature_dim: `0`
- feature_npz: ``
- val_feature_npz: ``
- test_feature_npz: ``
- target_schema: `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- type_vocab: `rectangular_notch`
- angle_encoding: `raw`
- continuous_targets_normalized: `False`
- type_class_weighting: `none`
- component_matching_mode: `fixed`
- export_predictions: `True`
- lambda_raster_bce: `0.0`
- lambda_raster_dice: `0.0`
- raster_loss_start_step: `0`
- raster_softness_cells: `1.0`
- raster_target_source: `masks`
- val_selection_metric: `none`
- val_selection_interval: `0`
- best_step: ``
- best_val_mask_iou: `nan`
- best_val_loss: `nan`
- lambda_center: `1.0`
- lambda_axis: `1.0`
- lambda_depth: `1.0`
- lambda_rotation: `1.0`
- lambda_center_grid: `0.0`
- lambda_center_axis_relative: `0.0`
- center_axis_relative_eps: `1e-06`
- center_representation: `continuous`
- center_bin_size_cells: `8`
- center_x_bins: `0`
- center_y_bins: `0`
- lambda_center_bin: `1.0`
- lambda_center_offset: `1.0`
- center_bin_x_weight: `1.0`
- center_bin_y_weight: `1.0`
- center_bin_slot_weights: ``
- center_bin_slot_weights_resolved: `1.0, 1.0, 1.0`
- aux_center_head: `False`
- lambda_aux_center_bin: `0.0`
- lambda_aux_center_offset: `0.0`
- aux_center_x_weight: `1.0`
- aux_center_y_weight: `1.0`
- lambda_type_extra: `0.0`
- lambda_rotation_extra: `0.0`
- rotation_loss_mode: `mse`
- signal_normalization: `per_sample_zscore`

## Continuous normalization

- `center_x`: mean=2200.27, std=880.337
- `center_y`: mean=1476.76, std=403.45
- `axis_x`: mean=692.667, std=149.045
- `axis_y`: mean=192.8, std=51.7387
- `depth_or_shape_param`: mean=77.1667, std=5.58022
- `rotation_angle`: mean=0, std=1

## Final metrics

- `train`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=2.099318e+02, param_mask_iou=3.811870e-02
- `val`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=1.977240e+02, param_mask_iou=7.817738e-02
- `test`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=2.136597e+02, param_mask_iou=3.644807e-02
