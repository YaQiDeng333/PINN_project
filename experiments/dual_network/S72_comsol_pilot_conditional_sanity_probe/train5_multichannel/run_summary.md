# Conditional supervised runner summary

- npz_path: `experiments/dual_network/S71_comsol_pilot_ingest/converted/comsol_multiheight_pilot.npz`
- sample_indices: `0,1,2,3,4`
- eval_npz_path: `None`
- eval_sample_indices: ``
- test_npz_path: `None`
- test_sample_indices: ``
- steps: `300`
- lr: `0.001`
- hidden_dim: `64`
- num_layers: `3`
- latent_dim: `32`
- original_signals_shape: `(5, 3, 200)`
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
- lambda_mask_bce: `1.0`
- lambda_mask_dice: `1.0`
- lambda_mu_mse: `0.0`
- mask_temperature: `50.0`
- signal_normalization: `per_sample_zscore`
- signal_feature_mode: `raw`
- signal_ablation: `False`

S50 uses supervised mask losses only. Weak-form / physics losses are not connected in this skeleton.
Signal normalization is applied before signal feature construction; optional signal ablation is applied to the constructed encoder input.
Point signal features are generated from the signals actually passed to the model.
Direct mask head mode trains BCE / Dice on mask probability instead of a mu-threshold-derived soft mask.

Final losses:
- total_loss: `6.364195e-01`
- bce_loss: `2.174608e-01`
- dice_loss: `4.189587e-01`
- mu_mse_loss: `9.451262e+04`
- batch_mean_iou: `4.723449e-01`

Train average metrics:
- defect_iou: `4.767923e-01`
- defect_area_pred: `4.019000e+03`
- mu_mse: `9.244401e+04`
- mu_mae: `2.460563e+02`

No model weights, checkpoints, arrays, or images were saved.