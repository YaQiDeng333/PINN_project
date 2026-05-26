# COMSOL parametric inverse run summary

- train_npz: `experiments/dual_network/S218_comsol_v3_geometry_normalized/converted/train_comsol_v3_hard_case_normalized.npz`
- val_npz: `experiments/dual_network/S218_comsol_v3_geometry_normalized/converted/val_comsol_v3_hard_case_normalized.npz`
- test_npz: `experiments/dual_network/S218_comsol_v3_geometry_normalized/converted/test_comsol_v3_hard_case_normalized.npz`
- train_targets: `experiments/dual_network/S219_comsol_v3_normalized_parametric_targets/train/parametric_targets.npz`
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
- lambda_center_grid: `0.1`
- lambda_center_axis_relative: `0.0`
- center_axis_relative_eps: `1e-06`
- center_representation: `bin_offset`
- center_bin_size_cells: `8`
- center_x_bins: `25`
- center_y_bins: `13`
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

- `center_x`: mean=-0.000884124, std=0.0156504
- `center_y`: mean=-0.00015495, std=0.00268967
- `axis_x`: mean=0.0123141, std=0.0026497
- `axis_y`: mean=0.00128533, std=0.000344925
- `depth_or_shape_param`: mean=77.1667, std=5.58022
- `rotation_angle`: mean=0, std=1

## Center bin representation

- `x_min`: -0.03999999910593033
- `x_max`: 0.03999999910593033
- `y_min`: -0.009999999776482582
- `y_max`: 0.009999999776482582
- `dx`: 0.00040201004126563144
- `dy`: 0.00020202019750469862
- `bin_width_x`: 0.0032160803301250515
- `bin_width_y`: 0.001616161580037589
- `center_x_bins`: 25
- `center_y_bins`: 13

## Final metrics

- `train`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=8.307487e-01, param_mask_iou=1.953827e-02
- `val`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=8.544369e-01, param_mask_iou=4.712739e-02
- `test`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=8.189740e-01, param_mask_iou=4.477056e-02
