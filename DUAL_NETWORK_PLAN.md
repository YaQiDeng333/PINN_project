# Dual Network Variational Alternating Plan

## 1. 本支线目标

本支线 `feature/dual-network-variational` 用于探索双网络变分交替方法，将磁标势场 `phi` 与材料磁导率场 `mu` 分离建模：

- `PhiNet` 表示磁标势或与探头磁场可微相关的势函数。
- `MuNet` 表示二维区域内的相对磁导率分布。
- 训练目标不是直接替换 main 主线，而是在隔离分支上验证“固定一个场、用弱形式约束更新另一个场”的可行性。

当前数据上下文来自 `data_generator_v2.py`：每个样本包含一维 lift-off 探头信号 `signals`、二维 `mu_maps`、缺陷类型与几何元数据，以及规则网格坐标 `x/y`。计划应优先兼容这套数据结构，不修改数据生成入口。

## 2. 与 main 主线的区别

main 主线当前更适合作为稳定基线：保留数据生成、基础训练与评估入口，避免引入仍在验证中的交替优化逻辑。

本支线的区别：

- 使用两个网络分别表达 `phi` 和 `mu`，而不是把反演目标压进单一 PINN 输出。
- `phi` 主要通过探头观测损失、边界条件和物理残差约束。
- `mu` 在固定 `phi` 后主要通过弱形式残差、正则项和先验约束更新。
- 训练流程采用交替优化，而不是一次性端到端联合训练。
- 实验脚本、模型文件、配置和结果目录都使用 `dual_network` 前缀，与 main 的训练和评估路径隔离。

## 3. PhiNet / MuNet 的基本结构

### PhiNet

建议职责：

- 输入：空间坐标 `(x, y)`，以及可选的样本条件编码 `z_signal`。
- 输出：标量势函数 `phi(x, y)`。
- 通过自动微分得到场量，例如 `B = -grad(phi)` 或需要的探头方向分量。
- 观测损失在 lift-off 线上比较预测探头信号与 `signals`。

建议结构：

- 坐标 MLP，支持 Fourier features 或 sinusoidal positional encoding。
- 条件式版本可增加一个 signal encoder，将一维探头信号编码成 latent，再与坐标拼接。
- 输出保持标量，避免在 `PhiNet` 内直接输出 `mu`。

### MuNet

建议职责：

- 输入：空间坐标 `(x, y)`，以及可选的样本条件编码 `z_signal` 或 `z_phi`。
- 输出：正值相对磁导率 `mu_r(x, y)`。
- 对输出施加正值约束，例如 `mu = mu_min + softplus(raw_mu)`，或预测 `log_mu` 后指数映射。
- 可加入平滑、TV、背景接近 `mu_bg`、缺陷稀疏等正则。

建议结构：

- 坐标 MLP 或轻量坐标条件网络。
- 初始阶段可先做每个样本独立优化，降低条件式批训练复杂度。
- 后续再扩展为共享网络加样本条件编码。

## 4. 固定 phi 后不能用探头数据损失直接更新 mu 的原因

探头数据损失通常形如：

```text
L_data = || H(phi)|probe - signal ||^2
```

其中 `H(phi)` 是从 `phi` 及其空间导数得到的探头观测预测。若 `phi` 已固定，`H(phi)` 也固定，`L_data` 对 `mu` 没有有效依赖：

```text
d L_data / d theta_mu = 0
```

因此，固定 `phi` 后继续用探头数据损失更新 `MuNet`，不会给 `mu` 提供正确梯度。即使代码里人为让 `mu` 进入观测算子，也容易造成物理含义不清的伪梯度：探头观测应由满足介质方程的场间接响应 `mu`，而不是在 `phi` 不变时让 `mu` 直接修补观测误差。

结论：固定 `phi` 时，探头数据损失可以作为监控指标，但不应作为 `MuNet` 的直接更新目标。

## 5. 固定 phi 后应使用弱形式残差更新 mu

固定 `phi` 后，`mu` 应通过磁静态控制方程的弱形式约束来更新。典型形式可写为：

```text
int_Omega mu grad(phi) · grad(v) dOmega = boundary/source terms
```

或等价地最小化弱残差：

```text
R_v(mu; phi) =
int_Omega mu grad(phi) · grad(v) dOmega - F(v)
```

对一组测试函数 `v_k`、有限元基函数、局部窗口权重函数或随机 test functions，优化：

```text
L_mu = sum_k |R_vk(mu; phi_fixed)|^2
      + lambda_smooth L_smooth(mu)
      + lambda_prior L_prior(mu)
      + lambda_pos L_pos(mu)
```

这样 `mu` 的梯度来自控制方程本身：

```text
d L_mu / d theta_mu != 0
```

该路线的核心是：`mu` 不是直接拟合 lift-off 探头信号，而是在固定场 `phi` 的条件下，使介质参数与弱形式物理平衡一致。

## 6. 交替训练流程

建议流程：

1. 数据准备：复用 main 的 `.npz` 数据格式，读取 `signals`、`mu_maps`、`x/y` 和元数据。
2. 初始化 `MuNet`：从背景值 `mu_bg` 开始，或用少量监督 `mu_maps` 进行 warm start。
3. 训练 `PhiNet`：固定 `MuNet`，最小化探头数据损失、边界条件损失和强/弱形式物理残差。
4. 训练 `MuNet`：固定 `PhiNet`，冻结 `phi` 和 `grad(phi)`，用弱形式残差更新 `mu`，并加入正值、平滑和先验正则。
5. 交替循环：重复 `phi` step 与 `mu` step，记录每轮的 data loss、weak residual、`mu` 正则和验证指标。
6. 收敛判断：若探头误差、弱残差和 `mu` 变化量同时趋稳，则停止；若两者震荡，降低其中一个 step 的学习率或减少交替步数。
7. 评估：比较预测 `mu` 与 `mu_maps`，但监督 `mu_maps` 首先作为诊断指标，不作为固定 `phi` 后的主要更新依据。

## 7. 风险点

- 可辨识性风险：单条 lift-off 信号对二维 `mu` 反演可能欠定，需要先验、边界条件或多观测增强。
- 弱形式退化：若 `grad(phi)` 局部过小，弱残差对 `mu` 的约束会变弱。
- 交替震荡：`PhiNet` 和 `MuNet` 互相追逐，可能导致 loss 降低但物理场不稳定。
- 尺度问题：背景 `mu_bg = 1000`、缺陷 `mu = 1` 的尺度差异很大，直接回归可能数值困难。
- 边界/source 定义不足：弱形式右端项 `F(v)` 必须和物理假设一致，否则 `MuNet` 会学到补偿项。
- 数据生成物理简化：当前数据生成器用解析式构造 noisy lift-off 信号，并不一定严格满足后续 PDE 约束。
- 评估误导：探头数据 loss 下降不代表 `mu` 反演正确，需要同时看 `mu` IoU、区域误差、边界位置和物理残差。

## 8. 最小实现路线

第一阶段只做最小闭环，不改 main 现有入口：

1. 新增 `dual_network_models.py`：定义 `PhiNet`、`MuNet` 和必要的坐标编码。
2. 新增 `dual_network_losses.py`：实现探头观测损失、弱形式残差、正值和平滑正则。
3. 新增 `train_dual_variational.py`：实现交替训练循环，显式区分 `phi_step` 和 `mu_step`。
4. 新增 `evaluate_dual_variational.py`：输出探头误差、`mu` 重建误差、弱残差和可视化。
5. 新增独立配置，例如 `configs/dual_network_variational.yaml`。
6. 结果保存到 `runs/dual_network_variational/`，不复用 main 的输出目录。
7. 先在极小样本和低分辨率网格上做过拟合测试，再扩展到 train/val/test。

本阶段只创建方案文档，不修改 `train_pinn.py`、`evaluate_pinn.py`、`data_generator_v2.py`，不训练，不提交 Git。

## 9. 与 main 分支如何保持隔离

- 所有实验文件使用 `dual_network` 或 `dual_variational` 前缀。
- 不修改 main 的训练、评估、数据生成入口，尤其不修改 `train_pinn.py`、`evaluate_pinn.py`、`data_generator_v2.py`。
- 不把实验结果、模型权重、缓存数据提交到 Git；保持 `.gitignore` 覆盖生成物。
- 从 main 合并时只吸收公共数据格式和基础工具变更，不把本支线交替训练逻辑反向写入 main。
- 若未来需要复用代码，先抽公共工具到明确的新模块，再通过 PR review 决定是否进入 main。
- 本支线的每次实现都应能通过 `git diff main...feature/dual-network-variational` 清晰看到实验边界。
