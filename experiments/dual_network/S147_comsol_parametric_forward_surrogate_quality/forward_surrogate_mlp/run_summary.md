# COMSOL parametric forward surrogate run summary

- steps: 5000
- lr: 0.001
- hidden_dim: 256
- num_layers: 4
- max_components: 3
- input_dim: 27
- output_dim: 600
- target_schema: center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle
- type_vocab: rectangular_notch, rotated_rect
- signal_normalization: train_zscore
- checkpoint_saved: false
- weights_saved: false

## Metrics

- train: signal_nrmse_raw=3.767854e-01, signal_corr=9.258671e-01, peak_abs_nrmse=9.827892e-02, signal_mse_norm=5.891078e-01
- val: signal_nrmse_raw=5.026852e-01, signal_corr=8.657639e-01, peak_abs_nrmse=1.380844e-01, signal_mse_norm=9.909756e-01
- test: signal_nrmse_raw=4.577952e-01, signal_corr=8.886174e-01, peak_abs_nrmse=9.483848e-02, signal_mse_norm=8.798878e-01
