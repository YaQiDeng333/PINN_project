# S101 COMSOL V2 background-collapse suppression probe

## 目的

S101 在真实 COMSOL V2 train / val / test 数据上测试 area calibration / foreground floor 是否能避免 `defect_area_pred=0`，并恢复非零 train / val / test IoU。

## 数据

- train NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- signals shape: train `[100,3,200]`, val `[20,3,200]`, test `[20,3,200]`
- flattened signal length: `600`

## 共同配置

- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `steps=3000`
- `lr=1e-3`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_head_mode=mu_threshold`
- `mask_source=mu_threshold`
- `signal_normalization=per_sample_zscore`
- `train_point_subsample=4096`
- `point_sampling_mode=random`
- `history_interval=250`

## 结果

| config | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| v2_baseline_with_history | train | 0.000000e+00 | 0.000000e+00 | 1.530510e+05 | 3.870879e+02 |
| v2_baseline_with_history | val | 0.000000e+00 | 0.000000e+00 | 1.531232e+05 | 3.871611e+02 |
| v2_baseline_with_history | test | 0.000000e+00 | 0.000000e+00 | 1.530382e+05 | 3.870759e+02 |
| area_ratio_mse | train | 0.000000e+00 | 0.000000e+00 | 1.206031e+05 | 3.301588e+02 |
| area_ratio_mse | val | 0.000000e+00 | 0.000000e+00 | 1.216602e+05 | 3.314853e+02 |
| area_ratio_mse | test | 0.000000e+00 | 0.000000e+00 | 1.214181e+05 | 3.312257e+02 |
| foreground_floor | train | 0.000000e+00 | 0.000000e+00 | 1.530908e+05 | 3.871435e+02 |
| foreground_floor | val | 0.000000e+00 | 0.000000e+00 | 1.531815e+05 | 3.872425e+02 |
| foreground_floor | test | 0.000000e+00 | 0.000000e+00 | 1.530952e+05 | 3.871555e+02 |

## training history 观察

- `v2_baseline_with_history` 最终 hard `batch_area_pred=0`，但 soft area 仍非零，最后一条 `pred_area_soft_mean=305.031982`，`true_area_mean=218.729996`。
- `area_ratio_mse` 最终 hard `batch_area_pred=0`，但 soft area 非零，最后一条 `pred_area_soft_mean=337.937378`，`true_area_mean=226.479996`。该组降低了 `mu_mse` / `mu_mae`，但没有把 hard threshold prediction 推过 `mu_threshold`。
- `foreground_floor` 最终 hard `batch_area_pred=0`，最后一条 `pred_area_soft_mean=297.730835`，`true_area_mean=211.440002`。由于 soft area 已高于 floor，后期 `area_loss=0`，未能恢复 hard foreground。

## 当前判断

- `batch_ratio_mse` 和 `foreground_floor` 都没有恢复非零 `defect_area_pred` 或非零 IoU。
- area loss 对 soft foreground 有影响，尤其 `area_ratio_mse` 降低了连续 `mu` 误差，但当前 hard mask 仍被 `mu_threshold` 输出路径卡住。
- 当前问题不是纯粹的 foreground area 总量不足，而是 soft defect / predicted `mu` 没有形成可越过 hard threshold 的定位结构。
- 下一步不应继续只调 `lambda_area_loss=10` 附近的面积项；应转向 direct mask + area loss、threshold calibration、boundary/localization loss，或重新设计 V2 的 staged curriculum。
