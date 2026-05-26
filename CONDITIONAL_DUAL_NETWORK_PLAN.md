# Signal-conditioned dual-network plan

## 1. 阶段动机

当前 `per-sample runner` 已经证明了几个关键事实：

- weak-form 双网络工程路径可行，`PhiNet` / `MuNet` 交替优化可以在 `.npz` 样本上跑通；
- 半监督 `BCE mask prior` 在 `20x10`、`40x20`、`80x40`、`100x50`、`200x100` 等已验证分辨率和样本规模下稳定优于 baseline；
- 但当前 runner 是每个样本单独优化一套 `PhiNet` / `MuNet`；
- 当前最强结果依赖 `mu_label` 或 `label_mask` 构造的半监督 prior，因此仍是 diagnostic upper bound；
- 若要形成能与主线 baseline 正式比较的候选模型，必须转向 `signal-conditioned model`：推理时只输入 `Bz signal + coords`，直接输出 `mu_pred` / `phi_pred`。

S48 的意义是把支线从“逐样本优化实验”推进到“可批量训练、可推理泛化的条件模型”阶段。

## 2. 当前 per-sample optimization 的限制

当前 `train_dual_variational.py` runner 的限制很明确：

- 每个样本单独训练，不能直接泛化到新样本；
- 推理成本高，因为新样本仍需要重新优化网络参数；
- 当前最强 200x100 结果依赖 `mu_label` mask prior；
- `BCE mask prior` 使用了 `mu_label < 500`，不能作为部署推理信息；
- per-sample 结果可以证明结构潜力和诊断上界，但不能直接作为主线替代模型。

因此，后续主线可比模型必须把 `Bz_meas` / `signals` 纳入输入，并学习一个跨样本共享的映射。

## 3. conditional model 目标

目标模型的输入：

- `signals` / `Bz_meas`
- `coords`

目标模型的输出：

- `mu_pred`
- optional `phi_pred`

训练时可以使用：

- `mu_label` / `mask loss`
- `weak-form loss`
- `data loss`
- `TV`
- `area prior`

推理时不使用：

- `mu_label`
- `label_mask`

这个边界是 S48 之后的核心判断标准：训练阶段可以先使用监督信号建立上界和稳定性，但推理接口必须是 label-free。

## 4. 建议结构

### BzEncoder

`BzEncoder` 输入 `signals`，输出每个样本的 latent vector：

- input: `[B, signal_len]`
- output: `[B, latent_dim]`

### ConditionalMuNet

`ConditionalMuNet` 输入 `coords + latent`，输出 `mu(x,y)`：

- coords input: `[N,2]` 或 `[B,N,2]`
- latent input: `[B,latent_dim]`
- output: `[B,N,1]`
- `mu` 使用 `sigmoid` 映射到 `[mu_min, mu_max]`

### ConditionalPhiNet

`ConditionalPhiNet` 输入 `coords + latent`，输出 `phi(x,y)`：

- coords input: `[N,2]` 或 `[B,N,2]`
- latent input: `[B,latent_dim]`
- output: `[B,N,1]`

### ConditionalDualNet

`ConditionalDualNet` 包装：

- `BzEncoder`
- `ConditionalMuNet`
- `ConditionalPhiNet`

`forward(signals, coords)` 返回：

- `latent`
- `mu`
- `phi`

## 5. 与主线关系

这是 `feature/dual-network-variational` 支线的新阶段，不修改 `main`。

后续只有在完成正式的 `train / val / test` 评估之后，`signal-conditioned dual network` 才具备与主线 baseline 比较的可能性。不能把当前 per-sample runner 的半监督结果直接写成主线替代结果。

当前 per-sample 阶段的合理结论是：

- weak-form 双网络路径工程可行；
- 半监督 `BCE mask prior` 上界稳定优于 baseline；
- 200x100 下 `area3_bce7` 是当前较稳的半监督配置；
- `128x4` 容量候选不够稳定，默认容量仍保持 `32x2`。

## 6. S48 当前范围

S48 只做以下工作：

- 创建 `CONDITIONAL_DUAL_NETWORK_PLAN.md`；
- 创建 `conditional_dual_models.py` 模型骨架；
- 创建 `smoke_test_conditional_dual_models.py`；
- 创建 `experiments/dual_network/S48_conditional_model_skeleton/summary.md`；
- 更新支线日志、阶段总结、术语、artifact index 和 README。

S48 不做：

- 不做正式训练；
- 不运行大规模实验；
- 不保存 checkpoint；
- 不声称效果；
- 不修改主线训练文件；
- 不把 conditional model 结果写成已验证替代方案。

下一步建议是 S49：创建 conditional runner 的数据接口 / batch loader，让 `signals + coords` 能以 batch 方式进入 `ConditionalDualNet`。
