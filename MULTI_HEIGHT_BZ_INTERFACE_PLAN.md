# Multi-height / COMSOL-style Bz Signal Interface Plan

## 1. 为什么需要 multi-height / multi-channel Bz

当前 conditional dual-network 已经从 per-sample optimization 进入 `signals + coords -> mu / phi` 的共享模型阶段。S56-S63 的结果说明模型确实会使用 `Bz signal`：zero / shuffled signal ablation 会显著降低 train IoU。但 held-out val/test IoU 仍然很弱。

单条一维 Bz 对二维缺陷边界可能信息不足。S57 的 signal normalization 只带来小幅变化，S58-S59 的 conditioning / encoder 结构主要提升 train 拟合，S60 的局部 Bz point feature 没有稳定改善 test，S63 的 `raw_abs_grad` 单条 Bz 派生特征也没有稳定改善 val/test 泛化。

因此下一阶段需要增加输入物理信息，而不是继续只在单条 Bz 上添加派生通道。multi-height / multi-channel Bz 允许模型看到不同 lift-off 高度、不同 field component 或不同 probe line 的信号，为解决单条 Bz 信息不足和 signal-to-shape ambiguity 做准备。

## 2. 数据 schema 约定

S64 起 conditional 数据接口支持两种 `signals` 形状。

单通道：

```text
signals shape [num_samples, signal_len]
```

多通道：

```text
signals shape [num_samples, num_channels, signal_len]
```

建议 `.npz` metadata：

- `signal_channels`
- `signal_channel_names`
- `lift_off_values`
- `field_components`
- `source_type`，例如 `synthetic` / `comsol`

这些 metadata 当前是建议字段，不是 S64 的硬依赖。S64 的强制字段仍然是 `signals`、`mu_maps`，以及 `coords` 或 `x` + `y`。

## 3. 第一版兼容策略

- 如果 `signals` 是 `[B,L]`，保持旧行为，输出给 runner 的 tensor 仍为 `[B,L]`。
- 如果 `signals` 是 `[B,C,L]`，按 channels-first 顺序 flatten 成 `[B,C*L]`。
- flatten 顺序为 `[channel0 all x, channel1 all x, ...]`。
- `infer_signal_len(dataset)` 对 3D signals 返回 `C*L`。
- `BzEncoder` 暂时不需要大改，仍接收展平后的 `[B, signal_len]`。
- 后续如果需要保留通道结构，可以再引入 Conv1d 多通道 encoder 或显式 channel embedding。

## 4. COMSOL 接口目标

未来 COMSOL 导出可包含：

- 多个 lift-off 高度的 `Bz`
- `Bx + Bz` 多分量
- 多条 probe line
- 对应 defect mask / `mu_map`
- 同一 `x/y` 网格和 metadata

目标是把 COMSOL-style 输出转换为 conditional runner 可读的 `.npz`，使后续模型能在推理时只依赖多通道 Bz signals + coords。

## 5. S64 当前范围

S64 只做接口和 smoke test：

- 支持 2D / 3D `signals` 输入；
- 对 3D signals 做 channels-first flatten；
- 在 batch 和 runner summary 中记录原始 shape、channel 数、每通道长度、flatten 后长度；
- 用 tempfile smoke test 验证 data utils 和 training runner 能跑通 multi-channel signals。

S64 不做正式训练，不调用 COMSOL，不生成 COMSOL 数据，不保存 checkpoint，也不声称效果改善。
