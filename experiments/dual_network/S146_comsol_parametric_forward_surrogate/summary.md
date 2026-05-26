# S146 COMSOL parametric forward surrogate

## 目的

S146 新增一个轻量 learned forward surrogate，用于学习 `geometry parameters -> multi-height Bz signal`。该模型只作为后续 forward consistency 的实验 referee，不替代 COMSOL forward solver。

## 模型结构

- `build_forward_geometry_vector(...)` 将 fixed-order component targets 展平为 geometry vector。
- 每个 component 包含：
  - `presence`
  - `type one-hot` 或 type probabilities
  - `continuous` geometry parameters
- `ParametricForwardSurrogate` 是 MLP，输入 geometry vector，输出 flattened Bz signal。
- 默认真实 V2 输出维度为 `600 = 3 * 200`。

## geometry vector schema

第一版保持 S113/S115 的 fixed component order，不做 permutation matching。对每个 slot 按以下顺序拼接：

1. `presence`
2. `type_vocab` 对应的 one-hot / probability vector
3. `target_schema` 中的 continuous parameters

当前支持 raw `rotation_angle`，也支持 refined `rotation_sin` / `rotation_cos`，因为 continuous vector 只按 schema 长度和顺序传入。

## signal normalization

Forward surrogate training 使用 `train_zscore`：

- train signals 计算 per-signal-dimension mean / std；
- train / val / test 共用 train stats；
- 训练 loss 使用 normalized signal MSE；
- metrics 同时报告 normalized MSE/RMSE、反归一化 `signal_nrmse_raw`、`signal_corr`、`peak_abs_nrmse` 和 per-channel nrmse/corr。

## 当前边界

- 这是 learned surrogate，不是 COMSOL forward consistency。
- 不保存模型权重或 checkpoint。
- S147 会在 V2 train/val/test 上验证该 surrogate 是否足够可靠；只有通过 quality gate 后才进入 inverse forward-consistency probe。

## 自评

- geometry vector schema 明确，且与 fixed-order parametric target route 保持一致。
- signal normalization 只使用 train split stats，避免 val/test 泄漏。
- S146 只提供模型和 runner skeleton，不声称 surrogate 已可用于 consistency。
