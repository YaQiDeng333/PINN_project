NEXT_STEP

## 当前状态

`CURRENT_BASELINE` 仍然是 v3_complex mask-only grid decoder boundary model + validation-selected probability threshold `0.90`。

当前项目目标是缺陷边界形状反演。baseline 决策应优先考虑 IoU、Dice、area_error、`pred_area=0`、small / low-signal 表现，以及预测 mask 是否能解释观测到的 Bz 信号。

当前最有价值的 pending candidate 是 `lambda_forward=0.10` 的 forward consistency。第 18.4 结果显示，相比当前 mask-only grid decoder baseline，它有明确正信号：

* IoU 和 Dice 提升；
* area_error 下降；
* center_error 改善；
* Bz residual / Bz MSE 明显下降；
* `pred_area=0` 没有明显恶化；
* small / low-signal 的 IoU 和 Dice 提升，但 area_error 仍需要 review。

该 candidate 尚未替换 `CURRENT_BASELINE`。是否替换 baseline 需要 review / decision 后再决定。

## 当前下一步

优先处理第 18.4 forward consistency `lambda_forward=0.10` 结果。

review 应检查：

* mask-to-Bz surrogate 是否可靠，且训练 candidate 时是否正确冻结；
* checkpoint selection 是否只使用 validation set；
* probability threshold 是否只由 validation set 选择，test set 是否只用于最终评估；
* metrics CSV 与 summary 是否一致；
* Bz residual 的计算与 baseline 对比是否一致；
* small / low-signal 以及 polygon / rotated_rect 的 trade-off 是否可接受。

如果 review 通过，再讨论是否将 forward consistency `lambda_forward=0.10` 提升为新的 `CURRENT_BASELINE`。

如果 forward consistency 不能替换 baseline，则进入新的 geometry-aware / physics-consistent inversion 阶段，而不是继续做 decoder 或 threshold 小修补。

## 不再继续的方向

不再继续 selection metric 细调、ensemble 变体、threshold trick、loss trick 或小 decoder patch。

不再继续 SDF v2、boundary head v2、coordinate refinement v2、hand-crafted Bz features、普通 U-Net-like decoder、shape-type conditional、star-convex、retrieval 等已停止方向的小修补。
