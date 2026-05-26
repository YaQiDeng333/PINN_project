# S93 COMSOL V2 small-label adaptation probe

## 目的

S93 针对 S87-S90 发现的 V2 small-label / multi_defect 问题，测试两类 runner 适配是否能改善真实 COMSOL V2 conditional model：

- `positive_balanced` point sampling：提高训练 subsample 中正类点出现比例；
- `pos_weighted_bce` / `focal_bce`：缓解 small-label class imbalance。

本阶段不保存模型权重、checkpoint、数组或图片。

## 数据

- train NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- val NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- test NPZ: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`
- train samples: `0..99`
- val / test samples: `0..19`
- signals shape: train `[100,3,200]`, val/test `[20,3,200]`
- flattened encoder length: `600`

## 共同配置

- `steps=3000`
- `lr=1e-3`
- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `signal_normalization=per_sample_zscore`
- `signal_feature_mode=raw`
- `mask_head_mode=mu_threshold`
- `train_point_subsample=4096`
- `point_sampling_mode=positive_balanced`
- `positive_fraction=0.5`
- `lambda_mask_bce=1.0`
- `lambda_mask_dice=1.0`
- `lambda_mu_mse=0.0`

## 结果

| config | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| balanced_bce | train | 0.000000e+00 | 0.000000e+00 | 1.829188e+05 | 4.260846e+02 |
| balanced_bce | val | 0.000000e+00 | 0.000000e+00 | 1.829731e+05 | 4.261410e+02 |
| balanced_bce | test | 0.000000e+00 | 0.000000e+00 | 1.828957e+05 | 4.260584e+02 |
| balanced_pos_weight5 | train | 0.000000e+00 | 0.000000e+00 | 2.341214e+05 | 4.837961e+02 |
| balanced_pos_weight5 | val | 0.000000e+00 | 0.000000e+00 | 2.341359e+05 | 4.838108e+02 |
| balanced_pos_weight5 | test | 0.000000e+00 | 0.000000e+00 | 2.341293e+05 | 4.838044e+02 |
| balanced_focal | train | 0.000000e+00 | 0.000000e+00 | 2.154341e+05 | 4.637995e+02 |
| balanced_focal | val | 0.000000e+00 | 0.000000e+00 | 2.154632e+05 | 4.638292e+02 |
| balanced_focal | test | 0.000000e+00 | 0.000000e+00 | 2.154368e+05 | 4.638029e+02 |

所有 metrics 均为 finite。

## 与 S85 big_multichannel_v2 对比

S85 `big_multichannel_v2`：

- train IoU = `3.023806e-01`
- val IoU = `2.593440e-01`
- test IoU = `2.768323e-01`
- train / val / test `defect_area_pred` 约为 `1229.54` / `1272.50` / `1281.60`

S93 三组均退化为 `defect_area_pred=0`，IoU 也为 `0`，明显差于 S85 baseline。

## Claude Code review

因为 S93 出现零预测面积，已调用 Claude Code review 检查 loss / sampling / label 对齐风险。review 结论：

- 没有发现 must-fix 实现错误；
- `coords` / `mu_label` / `mask_label` 采样对齐正确；
- 默认 `bce` / `random` 路径保持旧行为；
- 零预测面积更可能是训练动态问题，而不是实现 bug；
- `positive_balanced` 当前按 batch 内任一样本的正类点聚合选点，这是一个明确的设计取舍。

## 当前判断

- `positive_balanced` sampling 没有改善 V2 train fit，反而导致 full-grid 预测塌缩为全背景。
- `pos_weighted_bce` 没有恢复正类预测，且连续 `mu` 误差更差。
- `focal_bce` 也没有改善 val/test，仍然全背景。
- 当前 small-label 适配不能作为 V2 后续默认训练配置。

## 下一步建议

- 保留 S85 `big_multichannel_v2` 作为当前 V2 baseline。
- 不建议继续围绕当前 `positive_balanced` + `mu_threshold` 组合盲目扫 `pos_weight` 或 `focal_alpha`。
- 下一步应考虑更结构化的目标适配，例如 direct mask head、area calibration、boundary-aware loss、curriculum 数据，或 V1-like larger defect bridge samples。
