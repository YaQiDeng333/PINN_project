# COMSOL parametric inverse forward consistency run summary

- forward_pretrain_steps: 3000
- inverse_steps: 3000
- lambda_forward_consistency: 0.1
- forward_hidden_dim: 256
- forward_num_layers: 4
- forward_soft_input_augmentation: 0.15
- forward_surrogate_saved: false
- inverse_weights_saved: false
- checkpoint_saved: false

## Metrics

- train: mask_iou=5.947000e-01, type_acc=1.000000e+00, rotation_mae=4.358479e+00, forward_signal_nrmse=3.687188e-01, forward_signal_corr=9.289096e-01
- val: mask_iou=3.103259e-01, type_acc=6.333333e-01, rotation_mae=6.573701e+00, forward_signal_nrmse=6.664540e-01, forward_signal_corr=7.720926e-01
- test: mask_iou=3.954980e-01, type_acc=6.500000e-01, rotation_mae=7.106478e+00, forward_signal_nrmse=5.632792e-01, forward_signal_corr=8.318901e-01
