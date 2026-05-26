# Conditional supervised runner summary

- npz_path: `experiments/dual_network/S53_conditional_train_val_probe/data/training_data_train.npz`
- sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79`
- eval_npz_path: `experiments/dual_network/S53_conditional_train_val_probe/data/training_data_val.npz`
- eval_sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- steps: `2000`
- lr: `0.001`
- hidden_dim: `128`
- num_layers: `4`
- latent_dim: `64`
- lambda_mask_bce: `1.0`
- lambda_mask_dice: `1.0`
- lambda_mu_mse: `0.0001`
- mask_temperature: `50.0`

S50 uses supervised mask losses only. Weak-form / physics losses are not connected in this skeleton.

Final losses:
- total_loss: `4.863420e+00`
- bce_loss: `3.519091e-01`
- dice_loss: `9.999092e-01`
- mu_mse_loss: `3.511602e+04`
- batch_mean_iou: `0.000000e+00`

Train average metrics:
- defect_iou: `0.000000e+00`
- defect_area_pred: `0.000000e+00`
- mu_mse: `3.511602e+04`
- mu_mae: `3.516792e+01`

Eval average metrics:
- defect_iou: `0.000000e+00`
- defect_area_pred: `0.000000e+00`
- mu_mse: `3.892078e+04`
- mu_mae: `3.897649e+01`

No model weights, checkpoints, arrays, or images were saved.