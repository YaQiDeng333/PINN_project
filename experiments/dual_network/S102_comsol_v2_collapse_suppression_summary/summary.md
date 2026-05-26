# S102 COMSOL V2 collapse suppression summary

## S99 背景塌缩结论

S93 和 S97 都显示 V2 conditional training 容易进入全背景解：`defect_area_pred=0`，train / val / test IoU 均为 0。V1 pretrain 能在 V1 数据上拟合，但进入 V2 finetune 后仍会塌缩。因此当前首要诊断对象是 V2 的 area / mask output dynamics。

## S100 新增功能

`train_conditional_dual.py` 新增：

- `--lambda-area-loss`
- `--area-loss-mode none|batch_ratio_mse|foreground_floor`
- `--foreground-floor-ratio`

`training_history.csv` 新增：

- `area_loss`
- `pred_area_soft_mean`
- `true_area_mean`

默认 `lambda_area_loss=0.0` 且 `area_loss_mode=none`，保持旧行为。

## S101 结果

- `v2_baseline_with_history` 仍然全背景，train / val / test IoU 全为 0。
- `area_ratio_mse` 仍然全背景，train / val / test IoU 全为 0；但它降低了 `mu_mse` / `mu_mae`，说明 area calibration 对连续输出有影响。
- `foreground_floor` 仍然全背景，train / val / test IoU 全为 0；后期 soft predicted area 已高于 floor，导致 area loss 归零，无法继续推动 hard foreground。

## 与 S85 / S97 对比

- S85 `big_multichannel_v2` 仍是当前 V2 最佳历史 baseline：train / val / test IoU = `3.023806e-01` / `2.593440e-01` / `2.768323e-01`。
- S97 的 V2-only 和 V1-to-V2 curriculum 都全背景。
- S101 的 area loss 没有恢复 S85 baseline，也没有恢复非零 hard foreground。

## 当前瓶颈更新

1. hard `mu_threshold` 输出路径和 V2 small-label / multi_defect 训练动态仍是主要瓶颈。
2. 简单 area ratio / foreground floor 对 soft area 有影响，但不足以恢复 hard foreground。
3. 当前缺少定位和形状约束，area 总量约束不能单独解决 V2。
4. 下一步需要 direct mask + area loss、threshold calibration、boundary/localization loss，或更平滑的 V1-like / intermediate / V2-like curriculum。
5. 直接继续扩大 V2 数据仍不是优先项，除非同时调整 objective 或 curriculum。

## 下一步建议

- 优先测试 `mask_head_mode=direct` + area loss，判断是否绕开 hard `mu_threshold` 输出瓶颈。
- 增加 threshold calibration probe：记录 predicted `mu` 分布与 `mu_threshold=500` 的距离，判断是否只是 threshold margin 不足。
- 设计 boundary / localization objective，避免只约束面积而无法定位多缺陷形状。
- 如果继续生成数据，应加入 intermediate-area samples 和 V1-like larger defects，构成 staged curriculum，而不是只扩大 V2 multi_defect。
