# S99 COMSOL V2 background-collapse failure summary

## 目的

S99 汇总 S93 和 S97 的共同失败模式，明确当前阶段只处理 V2 conditional training 的 `background-collapse` / `area_pred=0` 问题。

## 失败现象

- S93 `balanced_bce`、`balanced_pos_weight5`、`balanced_focal` 三组均退化为全背景预测。
- S93 三组 train / val / test `defect_iou` 均为 `0.000000e+00`。
- S93 三组 train / val / test `defect_area_pred` 均为 `0.000000e+00`。
- S97 `v2_only_baseline_reproduce` 最终 `defect_area_pred=0`，train / val / test IoU 均为 0。
- S97 `v1_pretrain_v2_finetune` 中，V1 pretrain 阶段可以学到 V1，但进入 V2 finetune 后仍然塌缩为全背景。

## 当前判断

- 这不是 target/mask 定义问题。S87 已显示 V1/V2 的 `mu_maps < 500` 与 `masks > 0.5` 完全一致。
- 这也不是简单数据量问题。V2-only 和 V1-to-V2 curriculum 都能进入同样的全背景解。
- 当前首要问题是 V2 small-label / multi_defect 训练动态容易进入全背景解。
- 下一步需要显式约束 predicted foreground area，而不是继续盲目扫 `pos_weight`、`focal_alpha` 或只扩大数据。

## 下一步

- S100 在 `train_conditional_dual.py` 中增加 `area_loss_mode`、`lambda_area_loss` 和 `foreground_floor_ratio`。
- S101 在 V2 train / val / test 数据上测试 `batch_ratio_mse` 和 `foreground_floor` 是否能恢复非零 `defect_area_pred` 和非零 IoU。
