# COMSOL parametric inverse run summary

- train_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- train_targets: `experiments/dual_network/S113_comsol_parametric_targets/train/parametric_targets.npz`
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

## Final metrics

- `train`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=3.098074e-02, param_mask_iou=6.980716e-01
- `val`: presence_accuracy=1.000000e+00, type_accuracy_present=6.500000e-01, continuous_mae_mean=1.289384e+00, param_mask_iou=3.699078e-01
- `test`: presence_accuracy=1.000000e+00, type_accuracy_present=6.666667e-01, continuous_mae_mean=1.290734e+00, param_mask_iou=4.244624e-01
