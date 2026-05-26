# S131 COMSOL parametric set-matching stage summary

## S126 prediction export 结论

S126 已经让 `train_comsol_parametric_inverse.py` 支持 `--export-predictions`。当前可以导出：

- train / val / test 的 per-component prediction CSV；
- train / val / test 的 per-sample mask metrics；
- `presence_true` / `presence_prob` / `presence_pred`；
- `type_true` / `type_pred` / `type_correct`；
- center、axis、depth、rotation 的 true / pred / error；
- `pred_mask_iou`、`pred_dice`、`oracle_mask_iou`、`oracle_gap`、`target_area` 和 `pred_area`。

S126 使用 S115 raw MLP 配置复现了 baseline，train / val / test mask IoU 为 `6.980716e-01` / `3.699078e-01` / `4.244624e-01`，说明 export 不改变默认 fixed-order 训练行为。

## S127 grouped diagnostics 结论

S127 显示 presence 不是当前瓶颈。主要瓶颈仍是 held-out type / rotation / geometry generalization：

- slot 1 相对更弱，但不足以说明 slot/order ambiguity 是主因；
- rotation error bin 与 mask IoU 明显相关；
- worst samples 存在较大 oracle gap，说明模型预测仍明显低于 S117 oracle 上限。

## S129 set matching 结论

S129 对比 fixed-order 与 `permutation_min`：

- fixed train / val / test mask IoU = `6.980716e-01` / `3.699078e-01` / `4.244624e-01`；
- permutation train / val / test mask IoU = `6.773069e-01` / `1.787238e-01` / `2.462286e-01`。

`permutation_min` 在 held-out type accuracy、rotation MAE 和 mask IoU 上均低于 fixed reference。因此 loss-side set matching 不作为下一阶段主方向，当前 `center_x` fixed order 不是最主要限制。

## 当前路线判断

Parametric route 继续，因为 S117 oracle gate 通过，且 parametric route 避免了 dense V2 route 的全背景 / 全前景塌缩。但下一步不继续 slot/order 或 permutation matching，而是转向 differentiable rasterization，使 mask-level error 可以直接约束 geometry parameters。

## 自评

- S126-S130 结论已准确收束。
- 没有把 parametric route 表述为最终成功。
- 下一步明确指向 S132-S135 differentiable raster mask supervision。
