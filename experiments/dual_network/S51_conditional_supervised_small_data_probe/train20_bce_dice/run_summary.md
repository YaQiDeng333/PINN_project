# Conditional supervised runner summary

- npz_path: `experiments/dual_network/S51_conditional_supervised_small_data_probe/data/training_data_train.npz`
- sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- steps: `300`
- lr: `0.001`
- hidden_dim: `64`
- num_layers: `3`
- latent_dim: `32`
- lambda_mask_bce: `1.0`
- lambda_mask_dice: `1.0`
- lambda_mu_mse: `0.0`
- mask_temperature: `50.0`

S50 uses supervised mask losses only. Weak-form / physics losses are not connected in this skeleton.

Final losses:
- total_loss: `5.495827e-01`
- bce_loss: `7.375120e-02`
- dice_loss: `4.758315e-01`
- mu_mse_loss: `3.639882e+04`
- batch_mean_iou: `5.307977e-01`

No model weights, checkpoints, arrays, or images were saved.