# S78 COMSOL mask-source training probe

## 目的

S78 在 runner 中增加 `mask_source` 支持，并在 COMSOL geometry 数据上比较：

- `mask_source=mu_threshold`
- `mask_source=masks`

S77 已证明两种 target 在数据层完全一致，因此 S78 主要确认 runner 路径和训练表现没有隐藏差异。

## 配置

两组均使用 S74 converted train / val / test NPZ：

- train / val / test samples = 50 / 10 / 10
- signals shape = `[samples,3,200]`
- flattened signal length = 600
- `hidden_dim = 128`
- `num_layers = 4`
- `latent_dim = 64`
- `steps = 3000`
- `signal_normalization = per_sample_zscore`
- `mask_head_mode = mu_threshold`
- `train_point_subsample = 4096`

## 结果

| run | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| mu_threshold_reference | train | 5.401838e-01 | 2.801840e+03 | 6.992226e+04 | 1.827356e+02 |
| mu_threshold_reference | val | 4.041796e-01 | 2.955000e+03 | 8.421128e+04 | 2.139873e+02 |
| mu_threshold_reference | test | 4.047063e-01 | 2.955000e+03 | 8.593594e+04 | 2.157137e+02 |
| provided_masks | train | 5.403162e-01 | 2.772800e+03 | 6.906628e+04 | 1.809457e+02 |
| provided_masks | val | 4.011194e-01 | 2.919000e+03 | 8.274252e+04 | 2.100176e+02 |
| provided_masks | test | 3.888755e-01 | 2.919000e+03 | 8.465980e+04 | 2.119368e+02 |

所有 metrics 均为 finite。

## 判断

`mask_source=masks` 没有显著改善 train / val / test IoU。由于 S77 已证明两类 mask 完全一致，S78 的小幅差异应视为训练随机波动，而不是 target 定义差异。

当前判断：target/mask source 不是当前主要瓶颈。后续 COMSOL conditional 默认仍可使用 `mask_source=mu_threshold`；如果后续 COMSOL 数据只提供 masks，再使用 `mask_source=masks`。
