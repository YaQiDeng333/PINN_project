# S115 COMSOL parametric inverse training probe

## 数据

- train NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- train targets: `experiments/dual_network/S113_comsol_parametric_targets/train/parametric_targets.npz`
- val targets: `experiments/dual_network/S113_comsol_parametric_targets/val/parametric_targets.npz`
- test targets: `experiments/dual_network/S113_comsol_parametric_targets/test/parametric_targets.npz`

Signals 使用 per-sample zscore 后 flatten 为 `[B,600]`。

## 配置

- model: `ParametricInverseNet`
- steps: `3000`
- lr: `1e-3`
- hidden_dim: `128`
- latent_dim: `64`
- max_components: `3`
- loss: presence BCE + type CE + continuous SmoothL1
- continuous targets 使用 train split present components 的 mean / std 标准化训练，最终反标准化评估。
- rasterized mask IoU / Dice 只做评估，不反传。

## 结果

| split | presence_accuracy | type_accuracy_present | continuous_mae_mean | center_mae | axis_mae | rotation_mae | depth_mae | param_mask_iou | param_mask_dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.000000e+00 | 1.000000e+00 | 3.098074e-02 | 1.978452e-04 | 7.397077e-06 | 1.854690e-01 | 5.010099e-06 | 6.980716e-01 | 8.199347e-01 |
| val | 1.000000e+00 | 6.500000e-01 | 1.289384e+00 | 1.485341e-03 | 5.920604e-04 | 7.731843e+00 | 3.065882e-04 | 3.699078e-01 | 5.213624e-01 |
| test | 1.000000e+00 | 6.666667e-01 | 1.290734e+00 | 1.285206e-03 | 5.781414e-04 | 7.740396e+00 | 2.824507e-04 | 4.244624e-01 | 5.748063e-01 |

## 判断

- presence 可以稳定预测，train / val / test 都为 `1.0`。
- type 在 train 上可拟合，val/test 约 `0.65`，说明 component type 有初步泛化信号但还不稳。
- center / axis / depth 误差在 val/test 上为毫米级或更低量级；rotation 误差约 `7.7` degree，是当前较明显误差来源。
- 非可微 rasterized mask IoU 在 val/test 达到约 `0.37` / `0.42`，明显高于最近 dense V2 full-background / full-foreground failure runs，也高于 S85 `big_multichannel_v2` 的 V2 test IoU `0.2768`。
- 这不是 dense mask baseline，而是参数反演路线的首个小 probe；结果说明该路线比继续盲调 dense sparse-mask loss 更有希望。
