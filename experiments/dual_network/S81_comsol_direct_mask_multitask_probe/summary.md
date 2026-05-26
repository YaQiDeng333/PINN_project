# S81 COMSOL direct mask / multitask output head probe

## ??

S81 ??? COMSOL geometry-variation multi-height Bz ????? direct mask head ??? `mu_mse` multi-task loss????? conditional model ??????? `mu_pred -> mu_threshold mask` ????????

## ??

- train NPZ: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/train_comsol_multiheight.npz`
- val NPZ: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/val_comsol_multiheight.npz`
- test NPZ: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/test_comsol_multiheight.npz`
- signals shape: train `[50,3,200]`, val `[10,3,200]`, test `[10,3,200]`
- flattened encoder length: `600`
- train indices: `0..49`; val/test indices: `0..9`

## ??

?????`steps=3000`, `lr=1e-3`, `hidden_dim=128`, `num_layers=4`, `latent_dim=64`, `lambda_mask_bce=1.0`, `lambda_mask_dice=1.0`, `signal_normalization=per_sample_zscore`, `signal_feature_mode=raw`, `mask_source=mu_threshold`, `train_point_subsample=4096`, `mask_temperature=50.0`?

????????

- `mu_threshold_reference`: `mask_head_mode=mu_threshold`, `lambda_mu_mse=0.0`
- `direct_mu0`: `mask_head_mode=direct`, `lambda_mu_mse=0.0`
- `direct_mu1e-5`: `mask_head_mode=direct`, `lambda_mu_mse=1e-5`

## ????

| run | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| mu_threshold_reference | train | 5.373772e-01 | 2.969960e+03 | 7.397257e+04 | 1.993209e+02 |
| mu_threshold_reference | val | 3.988888e-01 | 3.158000e+03 | 8.831216e+04 | 2.287678e+02 |
| mu_threshold_reference | test | 3.905152e-01 | 3.158000e+03 | 8.999904e+04 | 2.304564e+02 |
| direct_mu0 | train | 5.394977e-01 | 2.764040e+03 | 1.502643e+05 | 3.669666e+02 |
| direct_mu0 | val | 4.063294e-01 | 2.922000e+03 | 1.497879e+05 | 3.668474e+02 |
| direct_mu0 | test | 3.930224e-01 | 2.922000e+03 | 1.582121e+05 | 3.752800e+02 |
| direct_mu1e-5 | train | 5.290736e-01 | 2.680080e+03 | 5.370302e+04 | 1.084524e+02 |
| direct_mu1e-5 | val | 4.030755e-01 | 2.830000e+03 | 6.056485e+04 | 1.266919e+02 |
| direct_mu1e-5 | test | 3.944752e-01 | 2.830000e+03 | 6.837009e+04 | 1.345050e+02 |

## ? S75/S78/S79 baseline ??

- S75 `big_multichannel` train / val / test IoU ?? `5.391816e-01` / `4.067505e-01` / `3.997817e-01`?
- S78 `mu_threshold_reference` train / val / test IoU ?? `5.401838e-01` / `4.041796e-01` / `4.047063e-01`?
- S79 ?? train-fit adaptation ??? train IoU ??? `0.70`???????? held-out IoU?
- S81 ? `direct_mu0` ? train IoU ???? `mu_threshold_reference` ?????val/test IoU ????????
- `direct_mu1e-5` ?????? `mu_mse` / `mu_mae`?? train IoU ???val/test IoU ???????? COMSOL baseline?

## ????

1. direct mask head ???? `mu_threshold` ?? mask ??????
2. ?? `mu_mse` multi-task ????? `mu` ?????????? held-out IoU ???
3. ???? `mu_mse` ???????????????`mu_threshold` baseline ?????
4. ???????????? / ???????? conditional model / conditioning ?????

## ?????

- ?? S82 ?? S74-S81 ??????
- ?? S83 V2 COMSOL geometry-variation ??????????????? defect type?rotation ? boundary irregularity?
