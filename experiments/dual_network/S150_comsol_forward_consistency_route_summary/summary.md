# S150 COMSOL forward consistency route summary

## S145 结论

S145 确认 direct physics feature concat 没有超过 S115 / S143 raw signal baseline。Physics features 有 train-side signal，但 `features_only` 和 `concat_latent` 都没有稳定改善 held-out mask IoU。因此下一步转向 learned forward surrogate / forward consistency。

## S146/S147 forward surrogate

S146 新增 geometry -> multi-height Bz learned forward surrogate。S147 在 V2 上训练后通过 quality gate：

| split | signal_nrmse_raw | signal_corr |
| --- | ---: | ---: |
| train | 3.767854e-01 | 9.258671e-01 |
| val | 5.026852e-01 | 8.657639e-01 |
| test | 4.577952e-01 | 8.886174e-01 |

该结果说明 surrogate 捕捉了主要 Bz waveform，可以作为诊断型 consistency referee，但仍不是 COMSOL solver。

## S148 forward consistency support

S148 新增 in-memory forward consistency runner：

- forward surrogate 只在内存中训练并冻结；
- 不保存 surrogate / inverse 权重；
- predicted geometry 送入 surrogate 前使用 straight-through hard presence/type；
- continuous geometry 使用 inverse 输出反归一化后的物理参数；
- consistency target 复用 forward split 的 train-zscore normalized signal。

Claude Code review 指出 3 个 must-fix，已全部修复并复核通过。

## S149 结果

S149 比较 parameter-only、`lambda_forward_consistency=0.1` 和 `1.0`：

- parameter-only val / test mask IoU = `3.699078e-01` / `4.244624e-01`。
- `lambda=0.1` val / test mask IoU = `3.103259e-01` / `3.954980e-01`。
- `lambda=1.0` val / test mask IoU = `2.477730e-01` / `3.392353e-01`。

`lambda=0.1` 对 rotation MAE 有局部改善，但没有改善 mask IoU；`lambda=1.0` 过强并明显退化。

## 当前判断

Learned forward consistency 值得保留为诊断工具，但当前版本不作为默认 training objective。当前最佳 parametric 配置仍是 S115 / S143 raw MLP / shared head / fixed-order parameter-only baseline。

## 下一步建议

- 如果继续 forward route，应先改进 surrogate / consistency 设计，例如 residual weighting、staged very-short consistency、或只对 selected geometry dimensions 施加 consistency。
- 如果目标是提高 held-out mask IoU，下一步更适合测试 physics-derived auxiliary prediction、type/rotation-specific supervision、或更强但受约束的 geometry target。
- 不建议继续盲扫 forward consistency lambda；当前结果显示简单加权 residual 会牺牲 mask IoU。

## 自评

- S145-S150 完成了 learned forward surrogate 和 consistency probe 的闭环。
- 没有保存权重、checkpoint 或图片。
- 结论保持为 diagnostic：forward surrogate 可用，但 current consistency objective 未改善 inverse mask IoU。
