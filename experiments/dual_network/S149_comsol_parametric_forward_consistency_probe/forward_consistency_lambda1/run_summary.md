# COMSOL parametric inverse forward consistency run summary

- forward_pretrain_steps: 3000
- inverse_steps: 3000
- lambda_forward_consistency: 1.0
- forward_hidden_dim: 256
- forward_num_layers: 4
- forward_soft_input_augmentation: 0.15
- forward_surrogate_saved: false
- inverse_weights_saved: false
- checkpoint_saved: false

## Metrics

- train: mask_iou=4.301722e-01, type_acc=1.000000e+00, rotation_mae=7.075432e+00, forward_signal_nrmse=3.657509e-01, forward_signal_corr=9.300855e-01
- val: mask_iou=2.477730e-01, type_acc=6.166667e-01, rotation_mae=7.157398e+00, forward_signal_nrmse=5.195281e-01, forward_signal_corr=8.566328e-01
- test: mask_iou=3.392353e-01, type_acc=5.833333e-01, rotation_mae=7.192984e+00, forward_signal_nrmse=5.306449e-01, forward_signal_corr=8.537060e-01
