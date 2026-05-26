# S95 COMSOL V2 small-label adaptation failure summary

## S93 失败模式

S93 测试了三组 small-label adaptation：

- `balanced_bce`
- `balanced_pos_weight5`
- `balanced_focal`

三组都退化为全背景预测：

- train / val / test `defect_iou` 均为 `0.000000e+00`；
- train / val / test `defect_area_pred` 均为 `0.000000e+00`；
- metrics 均为 finite，但没有形成有效 mask。

这说明当前 `positive_balanced` sampling 与 weighted / focal BCE 没有解决 V2 的训练问题。

## 当前最佳 baseline

当前最佳 V2 baseline 仍是 S85 `big_multichannel_v2`：

- train IoU = `3.023806e-01`
- val IoU = `2.593440e-01`
- test IoU = `2.768323e-01`

S93 三组均低于该 baseline。

## 当前判断

- target/mask 定义不是主因，S87 已确认 `mu_maps < 500` 与 `masks > 0.5` 完全一致；
- 简单 class imbalance loss 不是当前解法；
- 当前更像训练动态、任务难度和 curriculum 问题；
- V2 从 V1-like ellipsoid 任务跳到 small-label multi_defect / non-ellipsoid，任务分布变化过大。

## 下一步

- S96 增加 `training_history.csv`，记录训练动态；
- S96 增加可选 pretrain / finetune curriculum；
- S97 运行 V2-only baseline reproduce 与 V1 pretrain -> V2 finetune 对比。
