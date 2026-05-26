# Conditional supervised runner summary

- npz_path: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/train_comsol_multiheight.npz`
- sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49`
- eval_npz_path: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/val_comsol_multiheight.npz`
- eval_sample_indices: `0,1,2,3,4,5,6,7,8,9`
- test_npz_path: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/test_comsol_multiheight.npz`
- test_sample_indices: `0,1,2,3,4,5,6,7,8,9`
- steps: `3000`
- lr: `0.001`
- hidden_dim: `128`
- num_layers: `4`
- latent_dim: `64`
- original_signals_shape: `(50, 3, 200)`
- per_sample_signal_original_shape: `(3, 200)`
- flattened_signal_length: `600`
- signal_channels: `3`
- signal_length_per_channel: `200`
- signal_flatten_order: `channels_first`
- encoder_input_length: `600`
- conditioning_mode: `concat`
- encoder_type: `mlp`
- point_signal_mode: `none`
- mask_head_mode: `direct`
- mask_source: `mu_threshold`
- lambda_mask_bce: `1.0`
- lambda_mask_dice: `1.0`
- lambda_mu_mse: `1e-05`
- mask_temperature: `50.0`
- signal_normalization: `per_sample_zscore`
- signal_feature_mode: `raw`
- signal_ablation: `False`
- train_point_subsample: `4096`
- eval_batch_size: `8`

S50 uses supervised mask losses only. Weak-form / physics losses are not connected in this skeleton.
Signal normalization is applied before signal feature construction; optional signal ablation is applied to the constructed encoder input.
Point signal features are generated from the signals actually passed to the model.
Direct mask head mode trains BCE / Dice on mask probability instead of a mu-threshold-derived soft mask.

Final losses:
- total_loss: `1.124773e+00`
- bce_loss: `1.944444e-01`
- dice_loss: `3.876290e-01`
- mu_mse_loss: `5.426998e+04`
- batch_mean_iou: `5.301180e-01`

Train average metrics:
- defect_iou: `5.290736e-01`
- defect_area_pred: `2.680080e+03`
- mu_mse: `5.370302e+04`
- mu_mae: `1.084524e+02`

Eval average metrics:
- defect_iou: `4.030755e-01`
- defect_area_pred: `2.830000e+03`
- mu_mse: `6.056485e+04`
- mu_mae: `1.266919e+02`

Test average metrics:
- defect_iou: `3.944752e-01`
- defect_area_pred: `2.830000e+03`
- mu_mse: `6.837009e+04`
- mu_mae: `1.345050e+02`

No model weights, checkpoints, arrays, or images were saved.