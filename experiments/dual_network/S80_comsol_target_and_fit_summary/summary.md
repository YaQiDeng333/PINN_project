# S80 COMSOL target/mask and train-fit summary

## S77 target/mask 结论

COMSOL geometry train / val / test 数据中的 `mu_maps < 500` 与 provided `masks > 0.5` 完全一致：

- train / val / test avg mask IoU 均为 1.0
- area difference 均为 0
- mismatch count 均为 0

因此 S75 train IoU 偏低不是由 `mu_threshold` mask 与 provided masks 不一致导致。

## S78 mask_source 结论

`mask_source=mu_threshold` 与 `mask_source=masks` 的训练表现基本一致：

- `mu_threshold_reference` test IoU = `4.047063e-01`
- `provided_masks` test IoU = `3.888755e-01`

由于两类 mask 数据本身完全一致，S78 的差异应视为训练随机波动。后续可以继续默认使用 `mask_source=mu_threshold`。

## S79 train-fit 结论

S78 最佳 train IoU 低于 0.70，因此执行 S79。结果显示：

- `longer_steps` test IoU = `4.049452e-01`
- `bigger_subsample` test IoU = `3.955204e-01`
- `bce2_dice1` test IoU = `3.696685e-01`

三组都没有显著提升 train fit，也没有带来稳定 held-out 改善。简单增加 steps、增大 point subsample 或提高 BCE 权重不是主要解决方案。

## 当前瓶颈排序

1. target/mask：不是当前主要瓶颈。`mu_maps` 与 `masks` 完全一致。
2. train fitting：存在瓶颈，但不是简单 steps / subsample / BCE 权重能解决。
3. data size/diversity：仍是高优先级瓶颈。当前只有 50/10/10，且 `defect_type` 固定为 ellipsoid。
4. model/loss：需要继续排查。尤其是 mask-only BCE + Dice、`mu_threshold` 方式和当前 conditional MLP 表达是否足够。

## 下一步建议

- 不需要因为 mask source 改默认设置；继续使用 `mask_source=mu_threshold` 即可。
- 不建议继续单纯加 steps 或 subsample。
- 下一阶段优先做 target / loss 诊断：例如 direct mask head 的 COMSOL 版本、轻量 `mu_mse`、boundary-aware loss 或 area calibration。
- 同时准备扩大 COMSOL geometry 数据，并增加 defect type、旋转角、边界不规则度等几何多样性。
