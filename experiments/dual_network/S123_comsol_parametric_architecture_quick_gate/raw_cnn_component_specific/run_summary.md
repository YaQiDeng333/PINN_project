# COMSOL parametric inverse run summary

- train_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test_npz: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- train_targets: `experiments/dual_network/S113_comsol_parametric_targets/train/parametric_targets.npz`
- steps: `2000`
- lr: `0.001`
- hidden_dim: `128`
- latent_dim: `64`
- max_components: `3`
- encoder_type: `cnn1d`
- head_mode: `component_specific`
- target_schema: `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`
- type_vocab: `rectangular_notch, rotated_rect`
- angle_encoding: `raw`
- continuous_targets_normalized: `False`
- type_class_weighting: `inverse_freq`
- lambda_center: `1.0`
- lambda_axis: `1.0`
- lambda_depth: `1.0`
- lambda_rotation: `3.0`
- signal_normalization: `per_sample_zscore`

## Continuous normalization

- `center_x`: mean=6.20882e-12, std=0.0120388
- `center_y`: mean=-0.000254667, std=0.00447725
- `axis_x`: mean=0.00493334, std=0.000498888
- `axis_y`: mean=0.005864, std=0.000642628
- `depth_or_shape_param`: mean=0.00182917, std=0.000424489
- `rotation_angle`: mean=-2.96667, std=11.6418

## Final metrics

- `train`: presence_accuracy=1.000000e+00, type_accuracy_present=8.166667e-01, continuous_mae_mean=1.034491e+00, param_mask_iou=3.880497e-01
- `val`: presence_accuracy=1.000000e+00, type_accuracy_present=7.000000e-01, continuous_mae_mean=1.083696e+00, param_mask_iou=3.662237e-01
- `test`: presence_accuracy=1.000000e+00, type_accuracy_present=6.666667e-01, continuous_mae_mean=1.208580e+00, param_mask_iou=3.892232e-01
