NEXT_STEP

## 当前状态

`CURRENT_BASELINE` 已更新为 v3_complex mask-only grid decoder + forward consistency：

* `lambda_forward = 0.10`
* validation-selected probability threshold = `0.80`
* forward surrogate = `checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`

当前项目目标仍是缺陷边界形状反演。baseline 决策优先看 IoU、Dice、area_error、`pred_area=0`、small / low-signal 表现，以及预测 mask 是否更能解释观测到的 Bz 信号。

第 18.4 review 后，`lambda_forward=0.10` forward consistency 已从待定候选提升为新的 `CURRENT_BASELINE`。相比上一版 mask-only grid decoder baseline，它改善了 overall IoU、Dice、area_error、center_error 和 Bz MSE，且 `pred_area=0` 没有恶化。

需要同时保留的限制是：polygon area_error 轻微恶化，polygon / rotated_rect 精细边界圆斑化问题仍未根本解决。因此不能把本 baseline 理解为边界问题已经完成解决。

## 当前下一步

后续实验必须以新的 forward consistency baseline 为对照。

下一阶段如果继续推进，应围绕 polygon / rotated_rect 精细边界问题，或更严格的 geometry-aware / physics-consistent inversion 设计新阶段。新的实验必须回答：

* 预测 mask 是否更贴合真实边界；
* 预测 mask 是否更能解释 Bz 信号；
* IoU、Dice、area_error、`pred_area=0`、small / low-signal 是否优于新的 `CURRENT_BASELINE`；
* polygon / rotated_rect 的圆斑化是否真的减轻，而不是只有局部指标波动。

## 不再继续的方向

不再继续 `lambda_forward` 搜索，不做 forward consistency v2。

不再继续 selection metric 细调、ensemble 变体、threshold trick、loss trick、小 decoder patch、SDF v2、boundary head v2、coordinate refinement v2、hand-crafted Bz features、普通 U-Net-like decoder、shape-type conditional、star-convex、retrieval 等已停止方向的小修补。
