# Conditional supervised runner summary

- npz_path: `experiments/dual_network/S54_conditional_train_val_test_probe/data/training_data_train.npz`
- sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199`
- eval_npz_path: `experiments/dual_network/S54_conditional_train_val_test_probe/data/training_data_val.npz`
- eval_sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49`
- test_npz_path: `experiments/dual_network/S54_conditional_train_val_test_probe/data/training_data_test.npz`
- test_sample_indices: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49`
- steps: `3000`
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
- total_loss: `1.662823e-01`
- bce_loss: `3.845225e-02`
- dice_loss: `1.278301e-01`
- mu_mse_loss: `6.323530e+03`
- batch_mean_iou: `8.627310e-01`

Train average metrics:
- defect_iou: `8.627310e-01`
- defect_area_pred: `6.645000e+00`
- mu_mse: `6.321843e+03`
- mu_mae: `1.538956e+01`

Eval average metrics:
- defect_iou: `5.660991e-02`
- defect_area_pred: `8.300000e+00`
- mu_mse: `5.970452e+04`
- mu_mae: `6.911570e+01`

Test average metrics:
- defect_iou: `8.526794e-02`
- defect_area_pred: `7.320000e+00`
- mu_mse: `5.634353e+04`
- mu_mae: `6.527965e+01`

No model weights, checkpoints, arrays, or images were saved.