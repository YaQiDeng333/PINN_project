# S89 COMSOL signal interpretation probe skipped

## 跳过原因

S89 的条件是：如果 S88 显示明显 signal offset / background 或 delta-vs-absolute 风险，则运行最小 raw vs centered probe。

S88 结果显示：

- V2 train mean_abs_signal = `1.528793e-04`，V1 train mean_abs_signal = `4.143948e-05`，V2/V1 = `3.689`。
- V2 train mean_peak_abs_signal = `4.868779e-04`，V1 train mean_peak_abs_signal = `4.144736e-05`，V2/V1 = `11.747`。
- V2 offset/peak = `0.041`，未显示强 DC/background 主导。
- V2 lift-off monotonic_decay_fraction：train `0.92`，val `0.95`，test `0.90`，总体符合 lift-off 增大后 peak abs 衰减的预期。
- S85 已使用 `signal_normalization=per_sample_zscore`，训练路径已经做每样本均值和尺度归一化。

因此，本阶段不执行额外的 center-only S89 训练，以避免重复训练和引入新的 runner 改动。

## 当前判断

V2 低于 V1 的主要原因不优先指向简单 signal offset。更可疑的是：

1. V2 label area 明显小于 V1；
2. V2 target 是更复杂的 multi_defect / non-ellipsoid 组合；
3. V2 的 target / signal 生成语义与 V1 不同，当前 runner/loss 可能不适配；
4. V1 signal 本身接近常量，说明 V1/V2 不只是“同一任务更多数据”的关系。

## 下一步建议

在继续训练前，优先做 S90 阶段总结，并决定是否需要：

- V2 target / mask 与 V1 的任务定义对齐检查；
- V2 label area calibration；
- multi_defect 目标的 loss / runner 适配；
- 或重新生成更接近 V1 任务语义的 COMSOL V2 数据。
