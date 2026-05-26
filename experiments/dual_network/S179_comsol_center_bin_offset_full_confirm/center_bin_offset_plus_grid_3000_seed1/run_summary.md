# COMSOL parametric inverse run summary

- train_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- train_targets: `experiments/dual_network/S113_comsol_parametric_targets/train/parametric_targets.npz`
- seed: `1`
- steps: `3000`
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
- type_vocab: `rectangular_notch, rotated_rect`
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
- lambda_type_extra: `0.0`
- lambda_rotation_extra: `0.0`
- rotation_loss_mode: `mse`
- signal_normalization: `per_sample_zscore`

## Continuous normalization

- `center_x`: mean=6.20882e-12, std=0.0120388
- `center_y`: mean=-0.000254667, std=0.00447725
- `axis_x`: mean=0.00493334, std=0.000498888
- `axis_y`: mean=0.005864, std=0.000642628
- `depth_or_shape_param`: mean=0.00182917, std=0.000424489
- `rotation_angle`: mean=-2.96667, std=11.6418

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

- `train`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=1.141518e-01, param_mask_iou=7.161009e-01
- `val`: presence_accuracy=1.000000e+00, type_accuracy_present=6.333333e-01, continuous_mae_mean=1.367458e+00, param_mask_iou=5.429350e-01
- `test`: presence_accuracy=1.000000e+00, type_accuracy_present=6.500000e-01, continuous_mae_mean=1.180152e+00, param_mask_iou=5.813199e-01
