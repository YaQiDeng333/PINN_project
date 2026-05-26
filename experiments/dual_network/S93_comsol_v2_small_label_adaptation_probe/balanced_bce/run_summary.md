# Conditional supervised runner summary

- npz_path: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99`
- eval_npz_path: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- eval_sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- test_npz_path: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- test_sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- steps: `3000`
- lr: `0.001`
- hidden_dim: `128`
- num_layers: `4`
- latent_dim: `64`
- original_signals_shape: `(100, 3, 200)`
- per_sample_signal_original_shape: `(3, 200)`
- flattened_signal_length: `600`
- signal_channels: `3`
- signal_length_per_channel: `200`
- signal_flatten_order: `channels_first`
- encoder_input_length: `600`
- conditioning_mode: `concat`
- encoder_type: `mlp`
- point_signal_mode: `none`
- mask_head_mode: `mu_threshold`
- mask_source: `mu_threshold`
- mask_bce_mode: `bce`
- pos_weight: `1.0`
- focal_gamma: `2.0`
- focal_alpha: `0.25`
- lambda_mask_bce: `1.0`
- lambda_mask_dice: `1.0`
- lambda_mu_mse: `0.0`
- mask_temperature: `50.0`
- signal_normalization: `per_sample_zscore`
- signal_feature_mode: `raw`
- signal_ablation: `False`
- train_point_subsample: `4096`
- point_sampling_mode: `positive_balanced`
- positive_fraction: `0.5`
- eval_batch_size: `8`

S50 uses supervised mask losses only. Weak-form / physics losses are not connected in this skeleton.
Signal normalization is applied before signal feature construction; optional signal ablation is applied to the constructed encoder input.
Point signal features are generated from the signals actually passed to the model.
Direct mask head mode trains BCE / Dice on mask probability instead of a mu-threshold-derived soft mask.

Final losses:
- total_loss: `1.230528e+00`
- bce_loss: `3.630222e-01`
- dice_loss: `8.675054e-01`
- mu_mse_loss: `1.924264e+05`
- batch_mean_iou: `0.000000e+00`

Train average metrics:
- defect_iou: `0.000000e+00`
- defect_area_pred: `0.000000e+00`
- mu_mse: `1.829188e+05`
- mu_mae: `4.260846e+02`

Eval average metrics:
- defect_iou: `0.000000e+00`
- defect_area_pred: `0.000000e+00`
- mu_mse: `1.829731e+05`
- mu_mae: `4.261410e+02`

Test average metrics:
- defect_iou: `0.000000e+00`
- defect_area_pred: `0.000000e+00`
- mu_mse: `1.828957e+05`
- mu_mae: `4.260584e+02`

No model weights, checkpoints, arrays, or images were saved.