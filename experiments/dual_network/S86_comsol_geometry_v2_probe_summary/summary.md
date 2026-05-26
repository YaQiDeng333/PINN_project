# S86 COMSOL geometry V2 pilot summary

## S84 ingest 结论

S84 已完成 COMSOL geometry V2 fallback train / val / test 数据接入。原始 `comsol_geometry_variation_v2_exports/` 数据被复制到 S84 raw 目录，并通过 S67 converter 转换为支线可读的 multi-channel NPZ。

converted signals shape：

- train：`[100,3,200]`
- val：`[20,3,200]`
- test：`[20,3,200]`

conditional loader 可将 signals channels-first flatten 为 `[B,600]`，`infer_signal_len=600`，`ConditionalDualNet(signal_len=600)` forward 检查通过。

## S85 主要结果

| run | train IoU | val IoU | test IoU |
| --- | ---: | ---: | ---: |
| medium_multichannel_v2 | 2.307939e-01 | 2.107340e-01 | 2.048062e-01 |
| big_multichannel_v2 | 3.023806e-01 | 2.593440e-01 | 2.768323e-01 |

`big_multichannel_v2` 是当前 V2 中较好的配置，但 train / val / test IoU 仍明显低于 V1 S75。

## medium vs big

- `big_multichannel_v2` 在 train、val、test 上均优于 `medium_multichannel_v2`。
- `big_multichannel_v2` 的 train-test gap 只有约 `2.56e-02`，说明当前不是典型 train 过高而 held-out 崩溃。
- 两组 train IoU 都偏低，说明当前主要问题包含 train fit 或 target / signal 适配问题。

## 与 V1 S75 对比

V1 S75：

- medium train / val / test IoU = `5.225618e-01` / `4.088045e-01` / `3.961416e-01`
- big train / val / test IoU = `5.391816e-01` / `4.067505e-01` / `3.997817e-01`

V2 S85：

- medium train / val / test IoU = `2.307939e-01` / `2.107340e-01` / `2.048062e-01`
- big train / val / test IoU = `3.023806e-01` / `2.593440e-01` / `2.768323e-01`

V2 fallback 数据没有显示出比 V1 S75 更好的 val/test 潜力。V2 的几何更复杂，但当前训练拟合也下降，不能简单解释为“几何多样性提升失败”；更可能需要先排查 V2 数据目标、mask 生成、信号语义和 runner/loss 适配。

## 当前数据限制

- 样本量为 fallback 规模：train = `100`，val = `20`，test = `20`。
- 不包含 `ellipsoid`；实际类型为 `rectangular_notch` / `rotated_rect` multi_defect。
- magnetic parameters 固定，没有逐样本变化。
- `boundary_irregularity` 是 component distance bin proxy，不是真实 free-form roughness。
- 当前 Bz signal 使用 saved COMSOL models 重新评估的三 lift-off `Bz`，和 V1 数据分布不完全相同。

## 下一步建议

1. 先做 V2 target / signal 诊断，而不是直接继续扩大样本数。
2. 检查 V2 `mu_maps` / `masks` 是否一致，并与 V1 label area / target distribution 对比。
3. 检查 V2 signal scale、baseline Bz / delta Bz 语义、lift-off 定义和 probe line 是否与 runner 假设一致。
4. 如果 target / signal 无异常，再围绕 `big_multichannel_v2` 做 runner/loss 诊断。
5. 若后续 V2 train fit 明显提升，再考虑扩大 COMSOL V2 数据规模。

## 阶段判断

当前 V2 尚不足以支持“继续扩大 COMSOL V2 数据规模”的强结论。更稳妥的下一步是做 V2 数据语义和 target/mask 诊断，确认 train fit 偏低不是由目标定义、信号语义或 rasterization 差异造成。
