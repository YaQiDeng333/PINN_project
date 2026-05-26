# S75 COMSOL geometry conditional train/val/test probe

## 目的

S75 使用 S74 converted train / val / test NPZ 运行第一轮真实 COMSOL multi-height geometry-variation conditional supervised probe，判断当前 conditional runner 是否能在真实多高度 Bz 数据上形成非平凡 train / val / test 表现。

## 数据

- train NPZ: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/train_comsol_multiheight.npz`
- val NPZ: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/val_comsol_multiheight.npz`
- test NPZ: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/test_comsol_multiheight.npz`

数据规模：

- train samples = 50
- val samples = 10
- test samples = 10
- signals shape = `[samples,3,200]`
- flattened signal length = 600
- grid = 200 x 100
- coords points = 20000

## 配置

两组配置均使用：

- `signal_normalization = per_sample_zscore`
- `signal_feature_mode = raw`
- `mask_head_mode = mu_threshold`
- `lambda_mask_bce = 1.0`
- `lambda_mask_dice = 1.0`
- `lambda_mu_mse = 0.0`
- `mask_temperature = 50.0`
- `train_point_subsample = 4096`

`medium_multichannel`：

- `steps = 3000`
- `hidden_dim = 64`
- `num_layers = 3`
- `latent_dim = 32`

`big_multichannel`：

- `steps = 3000`
- `hidden_dim = 128`
- `num_layers = 4`
- `latent_dim = 64`

训练 loss 使用每步随机采样的 4096 个坐标点；最终 train / val / test metrics 使用完整 20000 个坐标点。

## 结果

| run | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| medium_multichannel | train | 5.225618e-01 | 2.740820e+03 | 7.717607e+04 | 2.152721e+02 |
| medium_multichannel | val | 4.088045e-01 | 2.874000e+03 | 8.694478e+04 | 2.331805e+02 |
| medium_multichannel | test | 3.961416e-01 | 2.874000e+03 | 8.931295e+04 | 2.355511e+02 |
| big_multichannel | train | 5.391816e-01 | 2.850320e+03 | 7.127827e+04 | 1.884028e+02 |
| big_multichannel | val | 4.067505e-01 | 3.006000e+03 | 8.602336e+04 | 2.203345e+02 |
| big_multichannel | test | 3.997817e-01 | 3.006000e+03 | 8.749112e+04 | 2.218037e+02 |

train-val gap：

- medium: `1.137573e-01`
- big: `1.324311e-01`

train-test gap：

- medium: `1.264202e-01`
- big: `1.393999e-01`

## 判断

两组实验均正常完成，metrics 全部 finite，且没有保存模型权重、checkpoint、`.npy` 或图片。

相对早期 synthetic single-Bz conditional 阶段约 0.1 左右的 held-out IoU，本次真实 COMSOL multi-height geometry-variation 数据在 val/test 上达到约 0.40，显示出更强的 held-out 潜力。`big_multichannel` 的 test IoU 和连续 mu 误差略好，`medium_multichannel` 的 val IoU 略好且 train-val gap 略小。

当前仍存在明显 train-held-out gap，但不属于“train 高、val/test 崩塌”的模式。下一步应优先扩大真实 COMSOL geometry 数据，并同时检查 target/mask 定义、loss balance 和 validation-aware selection。
