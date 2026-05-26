# S85 COMSOL geometry V2 conditional probe

## 目的

S85 使用 S84 converted train / val / test NPZ 运行第一轮真实 COMSOL geometry V2 fallback conditional supervised probe，判断更丰富的 multi_defect 几何变化是否相对 V1 S75 改善 held-out 泛化。

## 数据

- train NPZ：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val NPZ：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test NPZ：`experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- train / val / test samples：`100 / 20 / 20`
- signals shape：train `[100,3,200]`，val `[20,3,200]`，test `[20,3,200]`
- flattened signal length：`600`
- target：`mu_maps` + `masks`

## 配置

共同配置：

- `steps=3000`
- `lr=1e-3`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`
- `mask_temperature=50.0`
- `signal_normalization=per_sample_zscore`
- `signal_feature_mode=raw`
- `mask_head_mode=mu_threshold`
- `train_point_subsample=4096`

两组模型：

- `medium_multichannel_v2`：`hidden_dim=64`，`num_layers=3`，`latent_dim=32`
- `big_multichannel_v2`：`hidden_dim=128`，`num_layers=4`，`latent_dim=64`

## 结果

| run | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| medium_multichannel_v2 | train | 2.307939e-01 | 1.400800e+03 | 8.968431e+04 | 2.620035e+02 |
| medium_multichannel_v2 | val | 2.107340e-01 | 1.414300e+03 | 8.943614e+04 | 2.600751e+02 |
| medium_multichannel_v2 | test | 2.048062e-01 | 1.461950e+03 | 8.862718e+04 | 2.569233e+02 |
| big_multichannel_v2 | train | 3.023806e-01 | 1.229540e+03 | 7.104924e+04 | 2.257999e+02 |
| big_multichannel_v2 | val | 2.593440e-01 | 1.272500e+03 | 7.206851e+04 | 2.254308e+02 |
| big_multichannel_v2 | test | 2.768323e-01 | 1.281600e+03 | 7.157907e+04 | 2.254821e+02 |

## gap

- `medium_multichannel_v2` train-val IoU gap = `2.006996e-02`
- `medium_multichannel_v2` train-test IoU gap = `2.599776e-02`
- `big_multichannel_v2` train-val IoU gap = `4.304658e-02`
- `big_multichannel_v2` train-test IoU gap = `2.555836e-02`

## 与 V1 S75 对比

V1 S75 的主要结果：

- `medium_multichannel` train / val / test IoU = `5.225618e-01` / `4.088045e-01` / `3.961416e-01`
- `big_multichannel` train / val / test IoU = `5.391816e-01` / `4.067505e-01` / `3.997817e-01`

S85 V2 的 best held-out 结果来自 `big_multichannel_v2`：

- val IoU = `2.593440e-01`
- test IoU = `2.768323e-01`

因此 V2 fallback 数据没有优于 V1 S75；当前 V2 的 train / val / test IoU 均明显更低。

## 当前判断

- `big_multichannel_v2` 优于 `medium_multichannel_v2`，但仍低于 V1 S75。
- V2 没有出现明显 train 高、val/test 低的典型过拟合；train 本身也偏低。
- 主要问题更像是数据目标 / 信号分布 / target rasterization 与当前 runner 的适配问题，或模型/loss 对更复杂 multi_defect 目标拟合不足，而不是单纯扩大数据后泛化立即改善。
- 当前不能据此宣称 V2 几何多样性改善了 held-out 泛化。

## 下一步建议

- 先检查 V2 的 target mask、label area、field signal 语义和与 V1 的数据分布差异。
- 若继续使用 V2 数据，优先围绕 `big_multichannel_v2` 做小规模 runner/loss 诊断。
- 不建议直接扩大 V2 到更大规模，除非先确认当前 V2 target / signal 生成方式与 conditional runner 的任务定义一致。
