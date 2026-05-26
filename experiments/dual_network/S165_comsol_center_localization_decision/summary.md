# S165 COMSOL center-localization decision

## 阶段结论

S160-S165 将 S158/S159 的 oracle ablation 结论转化为 targeted diagnostic 和 quick fix：当前最大瓶颈是 component center localization，而不是 type、rotation、forward consistency 或 raster loss 权重。

## S161 diagnostics

Val/test center error 明显高于 train。Val / test 的 `center_l2_grid_mae` 为 `8.017750` / `6.998191`，`center_axis_relative_l2_mae` 为 `0.445062` / `0.394215`。Center error 与 mask IoU 在 val/test 上强负相关，Pearson 分别约 `-0.927` / `-0.911`，Spearman 分别约 `-0.902` / `-0.908`。这解释了 S158 中 `gt_center` 几乎关闭 baseline-to-oracle gap 的现象。

## S162 support

`train_comsol_parametric_inverse.py` 新增：

- `--lambda-center-grid`
- `--lambda-center-axis-relative`
- `--center-axis-relative-eps`

默认值保持旧行为。Center grid loss 使用 x/y spacing 把 meter error 换算成 grid-cell error；axis-relative loss 使用 GT full-width/full-height axis 做归一；两者都只对 present components 计算，不保存权重、checkpoint 或图片。

## S163/S164 results

S163 quick gate 中，`center_grid_loss` 相比同轮 1500-step reference 同时改善 val/test mask IoU 和 center error；`center_axis_relative` 退化。S164 3000-step confirm 显示 `center_grid_loss_3000` train / val / test mask IoU = `0.726483` / `0.469423` / `0.498874`，超过 S115 / S158 historical baseline 的 `0.698072` / `0.369908` / `0.424462`。

## 当前最佳配置

当前最佳 parametric 配置更新为 raw MLP / shared head / fixed-order + `lambda_center_grid=0.1`，3000 steps。它仍低于 oracle，但已经是本阶段内第一个稳定改善 val/test mask IoU 的 targeted change。

## 下一步建议

- 继续 center-localization route，但不要盲目加大 center lambda。
- 优先测试 center-bin classification + offset、signal-to-center auxiliary head 或 per-component peak-position alignment features。
- 若后续发现 center regression loss 能改善坐标误差但不能进一步转化为 mask IoU，应明确转向 center representation，而不是继续加大 lambda。
- 保留 `center_grid_loss` 作为新阶段 reference，但需要 stable repeat / seed check 才能升级为默认。

## 自评

S160-S165 完成了从 oracle bottleneck 到 targeted loss quick fix 的闭环；S164 通过后说明 center-aware loss 值得继续，但路线应从 simple regression weight 转向更稳的 localization representation。
