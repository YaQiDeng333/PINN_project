# S98 COMSOL curriculum summary

## S95 失败模式

S95 总结了 S93 的失败模式：`balanced_bce`、`balanced_pos_weight5` 和 `balanced_focal` 均退化为全背景预测，train / val / test IoU 均为 0，`defect_area_pred` 均为 0。因此 simple imbalance loss 和当前 `positive_balanced` sampling 不是 V2 的直接解法。

## S96 新增功能

S96 在 `train_conditional_dual.py` 中新增：

- `--history-interval`：按 step 写出 `training_history.csv`；
- `--pretrain-npz-path`；
- `--pretrain-sample-indices`；
- `--pretrain-steps`。

当没有 pretrain 或 `history_interval=0` 时，默认行为保持旧逻辑。pretrain 和 finetune 使用同一个模型和 optimizer，不保存中间权重。

## S97 结果

| config | train IoU | val IoU | test IoU | train area | val area | test area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2_only_baseline_reproduce | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| v1_pretrain_v2_finetune | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

V1 pretrain 本身能正常训练，pretrain step 2000 batch IoU 约为 `5.327341e-01`。但进入 V2 finetune 后仍快速塌缩为全背景。V2-only reproduce 也从非零预测面积逐步塌缩到零面积。

## curriculum 是否有效

当前 V1 -> V2 curriculum 没有改善 V2 train/val/test，也没有阻止全背景塌缩。

这说明当前问题不是简单 warm start 能解决的初始化问题。V2 finetune objective 会把模型重新推向全背景解。

## 当前瓶颈更新

1. V2 train dynamics：`mu_threshold` mask 输出在 V2 sparse positive target 上容易塌缩为全背景。
2. V2 multi_defect difficulty：V2 geometry 比 V1 更复杂，positive label area 更小。
3. 当前 curriculum / bridge data：单纯 V1 pretrain 后直接 V2 finetune 不足以稳定训练。
4. model/loss：需要更直接约束 mask area、边界或 positive coverage。
5. data diversity：后续数据应包含 V1-like bridge samples，但不能只依赖 V1 warm start。

## 下一步建议

- 如果继续 runner 方向：优先测试 area calibration、positive area prior、direct mask head 或 boundary-aware objective。
- 如果继续数据方向：准备 mixed curriculum，包括 V1-like larger-area ellipsoid、intermediate-area defects 和 V2-like multi_defect，而不是只做 V1 pretrain。
- 如果继续模型方向：考虑对 mask head、conditioning 或 point features 做结构调整，让模型更直接学习 small multi-component masks。
- 暂时不建议继续盲目扩大 V2 数据规模或继续扫 `pos_weight` / `focal_alpha`。
