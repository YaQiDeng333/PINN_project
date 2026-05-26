# S97 COMSOL V1-to-V2 curriculum bridge probe

## 目的

S97 比较 V2-only baseline 与 V1 pretrain -> V2 finetune，判断 V1-like COMSOL geometry 数据是否能帮助 V2 small-label / multi_defect 训练进入更好的解空间。

## 数据

- V1 pretrain: `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/train_comsol_multiheight.npz`
- V2 train: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`
- V2 val: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`
- V2 test: `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`

V1 train indices = `0..49`；V2 train indices = `0..99`；V2 val/test indices = `0..19`。

## 配置

共同配置：

- `hidden_dim=128`
- `num_layers=4`
- `latent_dim=64`
- `steps=3000`
- `lr=1e-3`
- `signal_normalization=per_sample_zscore`
- `signal_feature_mode=raw`
- `mask_head_mode=mu_threshold`
- `mask_source=mu_threshold`
- `train_point_subsample=4096`
- `point_sampling_mode=random`
- `history_interval=250`

两组：

- `v2_only_baseline_reproduce`: V2-only 3000 steps；
- `v1_pretrain_v2_finetune`: V1 pretrain 2000 steps，然后 V2 finetune 3000 steps。

## 结果

| config | split | defect_iou | defect_area_pred | mu_mse | mu_mae |
| --- | --- | ---: | ---: | ---: | ---: |
| v2_only_baseline_reproduce | train | 0.000000e+00 | 0.000000e+00 | 1.541616e+05 | 3.886333e+02 |
| v2_only_baseline_reproduce | val | 0.000000e+00 | 0.000000e+00 | 1.544101e+05 | 3.889507e+02 |
| v2_only_baseline_reproduce | test | 0.000000e+00 | 0.000000e+00 | 1.542379e+05 | 3.887432e+02 |
| v1_pretrain_v2_finetune | train | 0.000000e+00 | 0.000000e+00 | 1.578316e+05 | 3.936690e+02 |
| v1_pretrain_v2_finetune | val | 0.000000e+00 | 0.000000e+00 | 1.574977e+05 | 3.931907e+02 |
| v1_pretrain_v2_finetune | test | 0.000000e+00 | 0.000000e+00 | 1.583427e+05 | 3.943520e+02 |

所有 metrics 均为 finite。

## training_history 观察

`v2_only_baseline_reproduce`：

- first finetune history row: `batch_area_pred=6.963200e+02`, `batch_iou=9.438477e-03`；
- final finetune history row: `batch_area_pred=0.000000e+00`, `batch_iou=0.000000e+00`；
- 13 行 history 中 12 行为 `batch_area_pred=0`。

`v1_pretrain_v2_finetune`：

- V1 pretrain 能正常拟合，step 2000 的 batch IoU 约 `5.327341e-01`；
- V2 finetune step 1 仍有 `batch_iou=4.208252e-02`；
- V2 finetune 后续同样塌缩，final `batch_area_pred=0.000000e+00`；
- 22 行 history 中 12 行为 `batch_area_pred=0`。

## Claude Code review

因为 S97 history 显示全背景塌缩，已调用 Claude Code review。结论：

- 未发现 runner / curriculum / target alignment 的 must-fix bug；
- V1 pretrain 能正常收敛，说明模型、loss、optimizer 和训练 loop 可工作；
- V2-only 与 curriculum 都塌缩，说明问题是 V2-specific training dynamics；
- 可以继续记录该研究结论并提交。

## 当前判断

- V1 pretrain 没有改善 V2 train/val/test；
- V1 pretrain 不能阻止 V2 finetune 阶段塌缩为全背景；
- 当前 V2 困难不是单纯初始化问题，也不是缺少 V1-like warm start；
- 更可能来自 V2 sparse positive target、multi_defect geometry、`mu_threshold` 输出动态和当前 mask loss 的组合。

## 下一步建议

- 不建议继续只做 V1->V2 pretrain；
- 优先处理 V2 输出动态，例如 area calibration、direct mask head、boundary-aware objective 或 explicit positive-area constraint；
- 数据侧可以准备 curriculum bridge，但应包含 V1-like larger-area samples 与 V2-like multi_defect samples 的混合阶段，而不是只做 V1 pretrain 后直接 finetune V2。
