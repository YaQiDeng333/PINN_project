# S79 COMSOL train-fit adaptation probe

## 触发条件

S78 最佳 train IoU 为 `5.403162e-01`，低于 0.70，因此按规则执行 S79。由于 S77 显示 `mu_threshold` 与 provided `masks` 完全一致，且 S78 `mu_threshold_reference` 的 test IoU 略高，S79 使用 `mask_source=mu_threshold`。

## 配置

三组均基于 S78 big 配置：

- `hidden_dim = 128`
- `num_layers = 4`
- `latent_dim = 64`
- `signal_normalization = per_sample_zscore`
- `mask_head_mode = mu_threshold`
- `mask_source = mu_threshold`

变化项：

- `longer_steps`: `steps=6000`, `train_point_subsample=4096`
- `bigger_subsample`: `steps=3000`, `train_point_subsample=8192`
- `bce2_dice1`: `steps=3000`, `train_point_subsample=4096`, `lambda_mask_bce=2.0`, `lambda_mask_dice=1.0`

## 结果

| run | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| longer_steps | train | 5.392316e-01 | 2.943000e+03 | 7.066265e+04 | 1.782617e+02 |
| longer_steps | val | 4.066157e-01 | 3.140000e+03 | 8.651854e+04 | 2.138750e+02 |
| longer_steps | test | 4.049452e-01 | 3.140000e+03 | 8.813846e+04 | 2.154965e+02 |
| bigger_subsample | train | 5.374537e-01 | 2.835540e+03 | 7.249279e+04 | 1.858850e+02 |
| bigger_subsample | val | 4.070184e-01 | 2.981000e+03 | 8.742576e+04 | 2.166012e+02 |
| bigger_subsample | test | 3.955204e-01 | 2.981000e+03 | 8.876076e+04 | 2.179375e+02 |
| bce2_dice1 | train | 5.295864e-01 | 2.457080e+03 | 7.236016e+04 | 1.914966e+02 |
| bce2_dice1 | val | 4.052605e-01 | 2.536000e+03 | 8.651577e+04 | 2.228795e+02 |
| bce2_dice1 | test | 3.696685e-01 | 2.536000e+03 | 8.859055e+04 | 2.249564e+02 |

## 判断

三组都没有把 train IoU 提升到 0.70。`longer_steps` 的 test IoU 最高，但只比 S78 `mu_threshold_reference` 略高；`bigger_subsample` 没有明显收益；`bce2_dice1` 降低预测面积并伤害 test IoU。

当前判断：train fit 偏低不是简单由训练步数不足、4096 点采样过小或 BCE 权重不足单独导致。下一步不应只继续加 steps 或采样点，而应重点检查 target/mask 的物理定义、loss 形式、模型表达和数据分布。
