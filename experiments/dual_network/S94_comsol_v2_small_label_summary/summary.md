# S94 COMSOL V2 small-label adaptation summary

## S91 runner loss 支持

S91 在 `train_conditional_dual.py` 中新增了 `mask_bce_mode`：

- `bce`：默认路径，保持旧 BCE 行为；
- `pos_weighted_bce`：对正类 mask 点施加 `pos_weight`；
- `focal_bce`：使用 `focal_gamma` 和 `focal_alpha` 计算 focal BCE。

`metrics.csv` / `eval_metrics.csv` / `test_metrics.csv` 记录 `mask_bce_mode`，`run_summary.md` 记录 `mask_bce_mode`、`pos_weight`、`focal_gamma` 和 `focal_alpha`。

## S92 positive-aware sampling 支持

S92 新增 `point_sampling_mode`：

- `random`：默认路径，保持旧 `train_point_subsample` 行为；
- `positive_balanced`：在训练时从 `mask_label` 中按 batch 聚合正类坐标，尽量按 `positive_fraction` 采样正类和负类点。

该采样只影响训练 loss，eval / test 仍使用完整坐标。`run_summary.md` 记录 `point_sampling_mode` 和 `positive_fraction`。

## S93 结果

| config | train IoU | val IoU | test IoU | train area | val area | test area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S85 big baseline | 3.023806e-01 | 2.593440e-01 | 2.768323e-01 | 1.229540e+03 | 1.272500e+03 | 1.281600e+03 |
| balanced_bce | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| balanced_pos_weight5 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| balanced_focal | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

三组 S93 metrics 均为 finite，但全部退化为全背景预测。

## 当前瓶颈更新

当前瓶颈排序：

1. `mu_threshold` 输出路径在 V2 small-label / multi_defect 场景下对 sparse point supervision 很敏感。
2. 当前 `positive_balanced` 采样没有改善 full-grid mask，反而使 full-grid 输出塌缩。
3. 简单 `pos_weighted_bce` / `focal_bce` 不能解决 V2 small-label 问题。
4. V2 的 multi_defect / non-ellipsoid geometry complexity 仍是核心挑战。
5. target/mask 定义仍不是主要瓶颈，S87 已证明二者一致。

## 结论

S91/S92 的 runner 能力可以保留为诊断工具，但 S93 结果不支持将 `positive_balanced`、`pos_weighted_bce` 或 `focal_bce` 作为当前 V2 默认训练策略。

## 下一步建议

- 先回到 S85 `big_multichannel_v2` baseline，而不是沿 S93 配置继续扫参数。
- 若继续改 runner，优先测试更直接的 mask target 路径或 area calibration，而不是仅调整 BCE。
- 若继续生成数据，建议加入 V1-like larger-area / ellipsoid bridge samples，形成 curriculum，降低从 V1 到 V2 的任务分布跳变。
- 对 V2 multi_defect 可考虑后续模型/conditioning 改造，例如 richer point features、multi-scale encoder 或 boundary-aware objective。
