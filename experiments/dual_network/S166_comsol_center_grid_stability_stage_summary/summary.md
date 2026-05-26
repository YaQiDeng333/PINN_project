# S166 COMSOL center-grid stability stage summary

## 目的

S160-S165 显示 `lambda_center_grid=0.1` 是当前唯一稳定改善 val/test mask IoU 的 targeted change，但 S164 仍是 single-run full probe，不能直接升级为默认候选。本阶段只做 stability validation，不继续调 `lambda`，不重跑 `center_axis_relative`，也不进入 center-bin / auxiliary head 实现。

## 已知依据

- S158 oracle ablation 证明 `center_x` / `center_y` localization 是当前最大瓶颈。
- S161 显示 val/test center error 与 mask IoU 强负相关。
- S163 quick gate 中 `center_grid_loss` 同时改善 val/test；`center_axis_relative` 退化。
- S164 full probe 中 `center_grid_loss_3000` train / val / test mask IoU = `0.726483` / `0.469423` / `0.498874`。

## 本阶段边界

- S167 只增加 `--seed`，用于 reproducible repeat。
- S168 只运行 center-grid repeat，复用 S164 为 `existing_unrecorded`，新增 seed1 / seed2。
- 如果 seed1 同时低于 historical param-only val/test baseline，则跳过 seed2，直接判为 unstable。
- S169/S170 必须逐条列出 acceptance criteria 的 true/false。

## 自评

该阶段的信息密度高于继续盲扫 loss：它直接判断 S164 是否是可复现改进，而不是扩大搜索空间。
