# S90 COMSOL V2 data/signal diagnostic summary

## S87 target / label / defect distribution 结论

S87 比较了 V1 S74 和 V2 S84 的 train / val / test target、mask、label area 和 defect distribution。

主要结果：

- V1/V2 的 `mu_maps < 500` 与 provided `masks > 0.5` 均完全一致：
  - avg mask IoU = `1.0`
  - total mismatch count = `0`
- V2 train mean label area ratio = `5.355850e-02`
- V1 train mean label area ratio = `1.172090e-01`
- V2/V1 train label area ratio = `0.457`
- V2 label area 明显小于 V1，IoU 任务更容易受到小目标、边界误差和 area calibration 的影响。
- V1 是 `ellipsoid`，并且磁性参数变化；V2 是 `rectangular_notch` / `rotated_rect` multi_defect 组合，包含 rotation 和 boundary proxy，但 magnetic parameters 固定。

判断：

- target/mask 定义没有发现错误。
- V2 相比 V1 是不同且更难的任务分布，不只是“更多样本的同任务扩展”。
- V2 低于 V1 可以部分由 label area 更小、目标更复杂和 defect distribution 改变解释。

## S88 signal semantics 结论

S88 比较了 V1/V2 signals 的 scale、lift-off 通道关系和 offset。

主要结果：

- V1 train mean_abs_signal = `4.143948e-05`
- V2 train mean_abs_signal = `1.528793e-04`
- V2/V1 mean_abs_signal = `3.689`
- V1 train mean_peak_abs_signal = `4.144736e-05`
- V2 train mean_peak_abs_signal = `4.868779e-04`
- V2/V1 mean_peak_abs_signal = `11.747`
- V2 offset/peak = `0.041`，未显示强 DC/background 主导。
- V2 lift-off monotonic_decay_fraction：train `0.92`，val `0.95`，test `0.90`。
- V2 lift-off 通道总体符合 peak abs 随 lift-off 增大而衰减的预期。
- V1 val/test signals 接近常量，说明 V1/V2 signal 语义并不完全可比。

判断：

- V2 signal scale 与 V1 有明显差异，尤其 peak abs 大约高一个数量级。
- V2 没有强 offset 风险，且 S85 已使用 `per_sample_zscore`，因此简单 center-only preprocessing 不是优先方向。
- V1 signals 接近常量这一点提示：V1/S75 的较高 IoU 可能受目标分布或数据生成方式影响，不能直接作为 V2 应达到的同任务基准。

## S89 执行情况

S89 已跳过。

跳过原因：

- S88 未发现 V2 存在强 DC/background offset。
- V2 lift-off 通道总体合理。
- S85 已使用 `signal_normalization=per_sample_zscore`，已经去除每样本均值并标准化尺度。
- 继续跑 center-only 训练会重复消耗训练时间，且不直接针对当前最可疑问题。

## V2 低于 V1 的最可能原因排序

1. label area / defect distribution：V2 label area 约为 V1 train 的 `45.7%`，且目标从单一 `ellipsoid` 变为更复杂的 multi_defect / non-ellipsoid。
2. model/loss 不适配：当前 BCE + Dice + `mu_threshold` 输出对更小、更复杂、多组件目标的拟合不足，S85 train IoU 也偏低。
3. signal scale / signal semantics：V2 signal peak 比 V1 大约高 `11.7x`，但 S85 已做 `per_sample_zscore`，因此它是需要关注的背景因素，不是最直接的 offset 修正问题。
4. data size / diversity：V2 fallback 只有 train=100 / val=20 / test=20，样本仍少，但在 train fit 尚低之前，不宜直接扩大。
5. target/mask：不属于主要瓶颈；V1/V2 `mu_maps` 与 `masks` 完全一致。

## 下一步建议

- 不要立即扩大 COMSOL V2 数据。
- 先做 multi_defect / small-label target 的 runner/loss 适配，例如 area calibration、boundary-aware loss、positive sampling 或 focal-style BCE。
- 对比 V1/V2 的 label area 后，可考虑对 V2 做 mask-size-aware metric / loss 诊断。
- 如果继续使用 V2，优先沿 `big_multichannel_v2` 配置做小规模训练拟合诊断。
- 如果要重新生成数据，应让 V2 同时包含一部分 V1-like ellipsoid / larger-area samples，作为 curriculum 或 bridge set，避免任务分布一次性跳变过大。

## 阶段判断

V2 表现低于 V1 不是 target/mask 错误导致，也不优先是简单 signal offset 导致。更可能是 V2 任务分布显著更难：目标更小、多组件、非椭圆、信号尺度不同，而当前 conditional runner/loss 尚未适配。
