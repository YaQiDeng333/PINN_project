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
- feature_fusion_mode: `concat_latent`
- feature_dim: `58`
- feature_npz: `experiments/dual_network/S141_comsol_mfl_physics_features/train/physics_features.npz`
- val_feature_npz: `experiments/dual_network/S141_comsol_mfl_physics_features/val/physics_features.npz`
- test_feature_npz: `experiments/dual_network/S141_comsol_mfl_physics_features/test/physics_features.npz`
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
- signal_normalization: `per_sample_zscore`

## Continuous normalization

- `center_x`: mean=6.20882e-12, std=0.0120388
- `center_y`: mean=-0.000254667, std=0.00447725
- `axis_x`: mean=0.00493334, std=0.000498888
- `axis_y`: mean=0.005864, std=0.000642628
- `depth_or_shape_param`: mean=0.00182917, std=0.000424489
- `rotation_angle`: mean=-2.96667, std=11.6418

## Feature normalization

- `ch0_mean`: mean=-1.85664e-05, std=1.72157e-05
- `ch0_std`: mean=0.000201362, std=3.45997e-05
- `ch0_min`: mean=-0.000471379, std=5.15454e-05
- `ch0_max`: mean=0.000424877, std=6.65144e-05
- `ch0_peak_abs`: mean=0.000486233, std=5.39815e-05
- `ch0_peak_to_peak`: mean=0.000896256, std=8.67846e-05
- `ch0_argmax_x`: mean=-0.0339015, std=0.0154677
- `ch0_argmin_x`: mean=0.0324301, std=0.0176329
- `ch0_argmax_abs_x`: mean=0.0159317, std=0.03351
- `ch0_energy`: mean=4.2385e-08, std=1.41698e-08
- `ch0_abs_area`: mean=0.00016576, std=3.70436e-05
- `ch0_signed_area`: mean=-1.85664e-05, std=1.72157e-05
- `ch0_positive_peak_count`: mean=9.38, std=3.325
- `ch0_negative_peak_count`: mean=11.22, std=2.91746
- `ch0_half_abs_width`: mean=0.0206513, std=0.00854641
- `ch0_center_of_abs_mass`: mean=0.00121808, std=0.00233089
- `ch0_left_right_abs_balance`: mean=0.0076774, std=0.0745526
- `ch1_mean`: mean=-1.7819e-05, std=1.61241e-05
- `ch1_std`: mean=0.000188795, std=3.25022e-05
- `ch1_min`: mean=-0.000442698, std=4.99848e-05
- ... 38 additional features omitted from summary

## Final metrics

- `train`: presence_accuracy=1.000000e+00, type_accuracy_present=1.000000e+00, continuous_mae_mean=5.225517e-02, param_mask_iou=6.756908e-01
- `val`: presence_accuracy=1.000000e+00, type_accuracy_present=6.666667e-01, continuous_mae_mean=1.024911e+00, param_mask_iou=3.313752e-01
- `test`: presence_accuracy=1.000000e+00, type_accuracy_present=5.833333e-01, continuous_mae_mean=1.480355e+00, param_mask_iou=3.051455e-01
