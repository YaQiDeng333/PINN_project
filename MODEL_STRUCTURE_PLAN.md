# 基于变分场重构与弱形式材料更新的交替双网络 PINN 方法

本文档是 `feature/dual-network-variational` 支线的理论设计文档。当前支线只探索双网络变分交替方案，不直接替代 main 主线的单网络监督反演流程，也不把支线内容同步回 main。

当前阶段只做理论设计和支线文档整理，不修改 `README.md`，不修改 `train_pinn.py`、`evaluate_pinn.py`、`data_generator_v2.py`，不训练，不提交 Git。

## 1. 支线目标

本支线目标是探索一种基于变分场重构与弱形式材料更新的双网络 PINN 反演方案。它与 main 主线的定位不同：

- main 主线是 `Bz signal + 空间坐标 (x, y) -> mu map` 的单网络监督反演。
- 本支线不直接替代 main，而是在独立 worktree / branch 中验证另一种物理约束路径。
- 本支线把磁标量势场 `phi(x, y)` 与磁导率分布 `mu(x, y)` 拆成两个网络，并采用交替优化。

基本网络定位：

- `phi-Net` 输入空间坐标 `(x, y)`，输出磁标量势 `phi(x, y)`。
- `mu-Net` 输入空间坐标 `(x, y)`，输出磁导率分布 `mu(x, y)`。
- 训练时固定一个网络，更新另一个网络；不采用两个网络同时端到端联合训练作为第一版方案。

核心链路：

```text
探头 Bz 信号
-> phi-Net 重构 phi 场
-> 固定 phi 后构造弱形式残差
-> mu-Net 更新 mu 场
-> 交替迭代
```

该支线的价值在于验证：是否可以先通过变分能量和探头数据重构可解释的场 `phi`，再通过静磁方程弱形式约束更新材料参数 `mu`。

## 2. 原方案问题

原始方案中若出现如下形式，需要明确修正：

```text
L_mu = L_data(mu; phi_fixed) + beta TV(mu)
```

该写法的问题是：固定 `phi` 后，若探头预测量定义为：

```text
Bz_pred = -dphi/dy
```

则数据损失为：

```text
L_data = mean((Bz_pred - Bz_meas)^2)
       = mean((-phi_y(x, y_s) - Bz_meas)^2)
```

此时 `L_data` 只依赖固定的 `phi` 及其导数，不含 `mu`。因此：

```text
d L_data / d theta_mu = 0
```

这意味着固定 `phi` 后，不能用“探头数据损失”直接更新 `mu-Net`。这样做不会给 `mu-Net` 提供有效梯度；即便代码中人为让 `mu` 进入观测损失，也会把物理链路改成不清晰的伪梯度。

修正原则：

- `phi-Net` 阶段可以使用探头数据损失，因为 `Bz_pred = -phi_y` 直接依赖 `phi`。
- `mu-Net` 阶段不能再使用固定 `phi` 下的数据损失作为核心更新项。
- 固定 `phi` 后，`mu-Net` 应通过静磁场方程 `div(mu grad phi)=0` 的弱形式残差获得梯度。

## 3. 修正后的双网络结构

修正后的结构采用两个坐标网络：

### phi-Net

```text
input:  (x, y)
output: phi(x, y)
```

`phi-Net` 表达磁标量势。通过自动微分得到：

```text
phi_x = dphi/dx
phi_y = dphi/dy
Bz_pred = -phi_y(x, y_s)
```

其中 `y_s` 是探头测量线或 lift-off 采样线。

### mu-Net

```text
input:  (x, y)
output: mu(x, y)
```

`mu-Net` 表达二维磁导率场。输出应保持物理可行，例如正值约束、范围约束或经过归一化尺度映射。第一版可以先使用简单 MLP，后续再考虑条件编码或共享批量训练。

### 交替关系

第 `k` 轮迭代中：

```text
固定 mu_k      -> 更新 phi_{k+1}
固定 phi_{k+1} -> 更新 mu_{k+1}
```

两个网络使用独立 optimizer。`phi` 阶段冻结 `mu-Net`，`mu` 阶段冻结 `phi-Net`，避免两个网络同时追逐同一个数据损失造成不可解释补偿。

## 4. phi-Net 变分场重构

固定当前材料场 `mu_k` 后，训练 `phi-Net`。目标函数为：

```text
L_phi = L_energy + lambda_bc L_bc + lambda_data L_data
```

其中能量项为：

```text
L_energy = mean(0.5 * mu_k * (phi_x^2 + phi_y^2))
```

该项来自磁标势场的能量泛函，在固定 `mu_k` 时鼓励 `phi` 满足与材料分布一致的低能量场。

边界条件项：

```text
L_bc
```

用于约束 Dirichlet 边界、硬约束边界或后续可能加入的 Neumann 磁通边界。第一版建议先采用清晰可实现的 Dirichlet 或硬约束边界设定，避免边界物理含义不明。

探头数据项：

```text
L_data = mean((-phi_y(x, y_s) - Bz_meas)^2)
```

该项只在 `phi-Net` 阶段作为核心观测约束。因为 `Bz_pred = -phi_y` 直接依赖 `phi`，所以它能对 `phi-Net` 提供有效梯度。

`phi-Net` 阶段的实现边界：

- 固定 `mu_k`，不更新 `mu-Net` 参数。
- 对 `mu_k` 使用 detach 或关闭 `mu-Net` 参数梯度。
- 记录 `L_energy`、`L_bc`、`L_data`，避免只看总 loss。
- 先验证低分辨率网格上的 `phi` 场是否平滑、边界是否稳定、探头线 `Bz_pred` 是否能拟合测量。

## 5. mu-Net 弱形式材料更新

固定当前场 `phi_k` 后，训练 `mu-Net`。目标函数为：

```text
L_mu = L_weak + beta TV(mu) + gamma L_prior
```

其中核心项是弱形式残差：

```text
L_weak = mean_q | integral_Omega mu * grad(phi_k) dot grad(v_q) dOmega |^2
```

该项来自无源静磁场方程：

```text
div(mu grad phi) = 0  in Omega
```

在采用 Dirichlet 边界或硬约束 Dirichlet 边界时，测试函数 `v_q` 在 Dirichlet 边界取零。无源区域且没有显式 Neumann 磁通输入时，弱形式右端为：

```text
F(v_q) = 0
```

因此第一版弱残差可写为：

```text
R_q(mu; phi_k) = integral_Omega mu * grad(phi_k) dot grad(v_q) dOmega
L_weak = mean_q |R_q(mu; phi_k)|^2
```

如果后续采用已知 Neumann 磁通边界：

```text
mu grad(phi) dot n = g_N  on partial_Omega_N
```

则弱形式右端应改为：

```text
F(v_q) = integral_{partial_Omega_N} g_N * v_q ds
```

对应残差为：

```text
R_q(mu; phi_k) =
integral_Omega mu * grad(phi_k) dot grad(v_q) dOmega
- integral_{partial_Omega_N} g_N * v_q ds
```

测试函数 `v_q` 的第一版选择：

- 局部 compact-support test functions。
- 低分辨率局部窗口函数优先，用于验证 `mu-Net` 是否能获得非零且稳定的弱残差梯度。
- 后续可扩展为随机平滑测试函数或网格 / FEM 类基函数。

正则项：

```text
TV(mu)
```

用于抑制材料场毛刺和孤立噪点。

先验项：

```text
L_prior
```

可以包括背景接近、正值约束、范围约束、缺陷稀疏先验或与 `mu_maps` 的诊断性监督 warm start。注意：`mu_maps` 可以作为诊断和少量 warm start 的工具，但不应成为本支线固定 `phi` 后的核心训练目标。

`mu-Net` 阶段的实现边界：

- 固定 `phi_k`，不更新 `phi-Net` 参数。
- 对 `phi_k`、`phi_x`、`phi_y` 使用 detach，或关闭 `phi-Net` 参数梯度。
- 不使用固定 `phi` 下的探头数据损失更新 `mu`。
- 记录 `L_weak`、`TV(mu)`、`L_prior` 以及诊断用 `mu_maps` 指标。

## 6. 交替优化流程

建议第一版流程如下：

1. 准备网格坐标、探头线坐标 `y_s`、测量信号 `Bz_meas`，并读取诊断用 `mu_maps`。
2. 初始化 `mu_0`，可从背景常数场开始，也可用少量监督 warm start。
3. `phi_step`：固定 `mu_k`，训练 `phi-Net`，最小化 `L_phi`。
4. 冻结并缓存 `phi_k`、`grad(phi_k)`，用于弱形式积分。
5. `mu_step`：固定 `phi_k`，训练 `mu-Net`，最小化 `L_mu`。
6. 交替重复，记录每一轮 `L_phi`、`L_weak`、`TV(mu)`、探头线误差和 `mu_maps` 诊断指标。
7. 如果探头误差下降但 `mu_maps` 诊断明显变差，应判定可能存在 `phi/mu` 补偿性漂移。

伪流程：

```text
for outer_iter in range(K):
    freeze(mu_net)
    unfreeze(phi_net)
    optimize L_phi(phi_net; mu_k)

    phi_k = detach(phi_net(coords))
    grad_phi_k = detach(grad(phi_k, coords))

    freeze(phi_net)
    unfreeze(mu_net)
    optimize L_mu(mu_net; phi_k, grad_phi_k)
```

关键实现要求：

- `phi-Net` 和 `mu-Net` 使用独立 optimizer。
- `phi_step` 冻结 `mu-Net`。
- `mu_step` 冻结 `phi-Net`。
- 固定 `phi` 时必须 detach `phi` 或关闭 `phi-Net` 参数梯度，只让 `mu-Net` 接收弱残差梯度。
- 不把 main 的 `train_pinn.py` 改造成该流程；支线实现应使用独立脚本，例如 `train_dual_variational.py`。

## 7. 凸性与收敛性表述边界

本文档不声称“两个子问题严格凸，每步都有全局最优保证”。该说法过强，不适合神经网络参数化实现。

更稳妥的表述是：

在函数空间层面，固定 `mu` 时，能量泛函关于 `phi` 具有凸性；固定 `phi` 时，弱形式残差关于 `mu` 具有凸结构，TV 正则项也保持凸性。因此交替优化具有明确的变分动机和多凸优化解释。

但实际实现采用神经网络参数化后：

- 损失关于网络权重并不严格凸。
- 每个子步骤通常只能找到局部可接受解，而非全局最优解。
- 优化结果依赖初始化、学习率、测试函数族、边界条件和正则权重。
- 多凸性只能作为算法设计依据和稳定性来源，不能作为全局收敛保证。

因此，本支线的收敛性表述应限定为：

```text
该交替方法具有变分动机和多凸结构解释；
在神经网络参数化下，不声明每一步达到全局最优；
实验上通过 loss 分解、弱残差、探头误差和 mu_maps 诊断指标共同验证稳定性。
```

## 8. 后续实现计划

第一阶段只做最小支线闭环：

1. 保留 main 训练入口不动，不修改 `train_pinn.py`、`evaluate_pinn.py`、`data_generator_v2.py`。
2. 新增 `dual_network_models.py`，定义 `PhiNet` 和 `MuNet`。
3. 新增 `dual_network_losses.py`，定义 `L_phi`、`L_weak`、TV 和 prior。
4. 新增 `train_dual_variational.py`，实现外层交替循环。
5. 新增 `evaluate_dual_variational.py`，只评估支线输出，不改变 main 指标定义。
6. 第 8.1 先使用低分辨率局部 compact-support test functions，验证固定 `phi` 后 `mu` 是否能获得有效弱残差梯度。
7. 只在少量样本上做过拟合 / 梯度方向验证，再考虑完整数据集。
8. 所有输出使用 `dual_network` 或 `dual_variational` 前缀，避免和 main 的 checkpoints / results 混用。

实现前必须先解决的问题：

- 明确 `Omega`、探头线 `y_s`、Dirichlet 边界和可能的 Neumann 边界定义。
- 明确数值积分权重、网格间距和测试函数归一化。
- 明确 `mu` 的尺度参数化，避免背景 `mu≈1000` 与缺陷 `mu≈1` 的尺度差导致优化不稳定。
- 明确 `L_prior` 是纯先验、warm start 还是诊断项，避免把本支线退化成 main 的监督回归。

## 9. 当前不做的事情

当前阶段不做以下事情：

- 不修改 main 主线代码。
- 不修改 `README.md`。
- 不修改 `train_pinn.py`、`evaluate_pinn.py`、`data_generator_v2.py`。
- 不训练模型。
- 不生成 checkpoint。
- 不提交 Git。
- 不把支线内容同步进 main。
- 不声称该方法已经优于 main baseline。
- 不使用固定 `phi` 下的数据损失直接更新 `mu-Net`。
- 不声称每个交替子问题在神经网络参数空间中严格凸或有全局最优保证。

---

# 附录：main 同步来的第 7.19 模型结构优化方案（当前支线不采用）

本附录只保留同步上下文，不作为 `feature/dual-network-variational` 的执行计划。当前支线不修改 `README.md`、`train_pinn.py`、`evaluate_pinn.py` 或 `data_generator_v2.py`。

# 第 7.19 步：模型结构优化方案

本文件记录第 7.19 步的结构分析和第 7.20A / 7.20B 实施计划。本阶段只做方案设计，不修改模型代码，不启动训练。

## 1. 当前结构总结

### BzEncoder

当前 `train_pinn.py` 使用一维卷积编码 Bz signal：

```text
Bz signal [B, L]
→ Conv1d(1, 16, kernel=5) + GELU
→ Conv1d(16, 32, kernel=5) + GELU
→ AdaptiveAvgPool1d(16)
→ Flatten
→ Linear(32 * 16, 128) + GELU
→ Linear(128, latent_dim=64) + GELU
→ bz_latent [B, 64]
```

该结构能把一维漏磁信号压缩成一个全局 latent vector，但它主要表达整体信号特征，对局部缺陷边界、多个缺陷之间的细粒度对应关系表达有限。

### Fourier feature / 坐标编码

空间坐标 `(x, y)` 先归一化，然后使用 Fourier feature：

```text
sin(freq * x), cos(freq * x), sin(freq * y), cos(freq * y)
```

当前频率数量为 21，频率范围约为 1 到 10，因此坐标特征维度为 84。该编码能增强 MLP 对空间高频变化的表达能力，比直接输入 `(x, y)` 更适合边界反演。

### Bz latent 与坐标特征融合

当前融合方式是简单拼接：

```text
bz_latent:       [B, 64]
coord_features:  [N, 84]

bz_latent 扩展到 [B, N, 64]
coord_features 扩展到 [B, N, 84]
concat → [B, N, 148]
```

也就是说，每个空间点都共享同一个 Bz 全局 latent，再和该点坐标特征拼接，由 decoder 输出该点的归一化 μ。

### MLP decoder

当前 decoder 是一个普通 MLP：

```text
Linear(148, 128) + Tanh
Linear(128, 128) + Tanh
Linear(128, 64)  + Tanh
Linear(64, 1)
Softplus()
```

decoder 对每个坐标点独立输出 μ，没有显式的局部空间卷积、边界约束或多缺陷实例建模。

### 输出层与 μ 尺度

当前输出层是 `Linear(64, 1) + Softplus()`。Softplus 将归一化 μ 输出约束到 `(0, +∞)`，提供下界但没有上界。

训练目标是：

```text
mu_target_norm = mu_map / 1000
```

因此背景约为 `1.0`，缺陷约为 `0.001`。预测后再乘以 `MU_SCALE = 1000` 恢复到真实 μ_r 尺度。

当前问题不是“完全没有约束”，而是 Softplus 对缺陷端的校准不够理想。缺陷区域目标接近 `mu_norm≈0.001`，但模型预测容易停留在 `mu_norm≈0.2-0.4`，对应真实 `μ_r≈200-400`。要让 Softplus 输出逼近 `0.001`，pre-activation 需要变得很负，这可能带来梯度较弱和输出偏软的问题。

当前标准 mask 阈值仍是：

```text
mu_r < 500
```

对应归一化阈值：

```text
mu_norm < 0.5
```

## 2. 当前问题定位

### small polygon 已能检出，但 μ 值仍偏软

第 7.12 到第 7.17 的实验说明，weighted MSE、soft Dice Loss 和 area-aware loss 已经能把 small polygon 从完全漏检中拉出来。尤其 soft Dice Loss 后，small polygon 的 `pred_area=0` 可以降到 0/25。

但第 7.18 的 threshold 分析显示，模型预测出来的缺陷 μ 往往停在 `μ_r≈200-400`，而不接近真实缺陷 `μ_r≈1`。这说明模型已经知道某些区域“像缺陷”，但输出校准不足，没有把缺陷区域压到足够低的 μ。

### threshold=300 能显著降低 area_error 的原因

标准阈值 `mu_r < 500` 下，许多边界周围的软预测也会被判为缺陷，因此 `pred_area` 系统性偏大。第 7.18 中把 threshold 降到 300 后，area_error 明显下降，说明大量过预测区域集中在 `300-500` 这个软边界区间。

这不是单纯的后处理问题。它反映的是模型输出分布没有被很好校准：真实缺陷区域没有稳定接近 `μ≈1`，背景到缺陷的过渡也偏宽。

### 继续调 loss 的收益递减

第 7.12 到第 7.17 已经尝试了：

1. weighted MSE；
2. soft Dice Loss；
3. symmetric area-aware loss；
4. over-only area loss；
5. defect_weight 和 lambda_area 组合验证。

这些实验已经覆盖了当前主要 loss 方向。结果是 small polygon 检出改善明显，但 polygon area_error 和输出校准仍没有根本解决。继续细扫 loss 权重很可能只是在 IoU、Dice、area_error、center_error 之间做局部折中，难以改变 μ 输出偏软的问题。

### 更可能的瓶颈

当前问题更可能由多个因素共同造成：

1. **输出层校准不足**：当前 Softplus 有下界但无上界，缺陷端要逼近 `mu_norm≈0.001` 时需要很负的 pre-activation，模型容易停在中间软输出。
2. **μ 输出参数化方式不合适**：直接回归 `0.001-1.0` 的归一化 μ，容易产生折中预测。
3. **decoder 表达能力有限**：当前 decoder 较浅较窄，对 polygon 边界、小目标和多个缺陷的细节表达不足。
4. **BzEncoder 不是第一优先瓶颈**：模型已经能检出 small polygon，说明 BzEncoder 至少能提供有用信号。第一步更应该处理输出校准和 decoder 表达能力。

## 3. 推荐结构优化方案

推荐第一个结构优化主方案为分阶段验证：

```text
第 7.20A：只做输出 μ 参数化校准
第 7.20B：若 7.20A 有效或部分有效，再轻量增强 decoder
```

核心思想是不要一次性同时改变输出参数化和 decoder 容量。第一步只在 decoder 输出端加入更符合物理范围的 μ 参数化，让模型更容易把缺陷区域压到接近 `μ≈1`，而不是停在 `μ≈200-400`；只有当该假说被验证后，再考虑增强 decoder。

### 建议的新输出参数化

保留 BzEncoder 和 Fourier feature，新增一个可选模型变体，例如：

```text
--model-variant baseline / calibrated_mu
```

默认仍为 `baseline`，保证旧流程不受影响。

在 `calibrated_mu` 中，保持当前 decoder 主体结构不变，即 `128 / 128 / 64 + Tanh`。decoder 最后一层先输出一个 logit，再通过 soft defect probability 构造归一化 μ：

```text
defect_prob = sigmoid(logit)
mu_norm = mu_min_norm + (1 - mu_min_norm) * (1 - defect_prob)
```

其中：

```text
mu_min_norm = 1 / MU_SCALE = 0.001
```

这样：

```text
defect_prob → 1  时，mu_norm → 0.001
defect_prob → 0  时，mu_norm → 1.0
```

该方式不是硬 clamp，仍然可微，但会把输出限制在物理合理范围内。它也让 decoder 的任务更接近“预测缺陷概率后映射为 μ”，有利于测试“输出校准不足”这一假说。

与当前 Softplus 相比，`calibrated_mu` 的重点不是简单增加约束，而是用 defect probability 的语义重新参数化输出空间。当前 Softplus 需要通过很负的 pre-activation 才能逼近 `mu_norm≈0.001`；而 calibrated_mu 直接把高 defect probability 映射到低 μ 区间，更适合验证缺陷端校准是否是主要瓶颈。

### decoder 轻量增强

decoder 增强不放在第 7.20A 中。只有当第 7.20A 的 calibrated μ 输出有效或部分有效后，第 7.20B 再考虑把 decoder 从当前 128/128/64 稍微增强，例如：

```text
Linear(input, 256) + SiLU
Linear(256, 256) + SiLU
Linear(256, 128) + SiLU
Linear(128, 64)  + SiLU
Linear(64, 1)
```

这仍然是 MLP，不引入复杂新模块，不改变数据集，也不改变评价指标。增强 decoder 的目的，是提高 polygon 边界和多缺陷空间细节的表达能力。它必须作为第 7.20B 单独验证，避免和输出参数化混在一起。

### 为什么不优先改 BzEncoder

当前 small polygon 已经能被检出，说明一维 Bz signal 的全局编码并非完全失效。BzEncoder 当然可能限制多缺陷定位，但如果第一轮结构实验同时改 BzEncoder、decoder 和输出层，将很难判断改善来自哪里。

因此第 7.20A 应优先只做输出参数化，保持 decoder 容量不变。第 7.20B 再视 7.20A 结果决定是否增强 decoder。

## 4. 第 7.20A / 7.20B 实施计划

### 阶段名称

第 7.20A 步：输出 μ 参数化校准实验。

第 7.20B 步：decoder 轻量增强实验，仅在第 7.20A 有效或部分有效后进行。

### 允许修改的文件

1. `train_pinn.py`
2. `evaluate_pinn.py`，仅在需要兼容新 checkpoint 的模型变体加载时做最小修改
3. `README.md`
4. `PINN优化路线.md`
5. `NEXT_STEP.md`
6. `EXPERIMENT_LOG.md`
7. `CURRENT_BASELINE.md`，只记录实验，不默认切换 baseline

### 不修改的内容

1. 不修改 `data_generator_v2.py`
2. 不修改评价指标定义
3. 不启用 physics_loss
4. 不启用 L-BFGS
5. 不加入 focal loss
6. 不加入 oversampling
7. 不覆盖旧 checkpoints
8. 不切换全项目 baseline

### 第 7.20A：输出 μ 参数化校准

第 7.20A 只改输出 μ 参数化，保持当前 decoder 主体结构不变：

```text
Linear(input, 128) + Tanh
Linear(128, 128) + Tanh
Linear(128, 64)  + Tanh
output head
```

建议新增：

```text
--model-variant baseline / calibrated_mu
```

新增模型类可命名为：

```text
PINNCalibratedMu
```

或在现有 `PINN` 中通过 `model_variant` 控制输出参数化。为了减少重复代码，推荐复用 BzEncoder、Fourier feature 和当前 decoder 主体，只替换 output head。

第 7.20A 的目标是单独验证：

```text
有界输出 mu_norm ∈ [0.001, 1.0] 是否能改善 μ 校准和 area_error
```

### 第 7.20B：decoder 轻量增强

如果第 7.20A 有效或部分有效，再进入第 7.20B。第 7.20B 才考虑 decoder 增强，例如：

```text
Linear(input, 256) + SiLU
Linear(256, 256) + SiLU
Linear(256, 128) + SiLU
Linear(128, 64)  + SiLU
output head
```

第 7.20B 的目标是验证更强 decoder 是否进一步改善 polygon 边界和 multi_defect 定位。不要在第 7.20A 中同时改变 decoder 容量，否则无法判断改善来自输出参数化还是 decoder 表达能力。

### 数据集

使用正式 v4 数据集：

```text
data/training_data_v4_balanced_complex_train.npz
data/training_data_v4_balanced_complex_val.npz
data/training_data_v4_balanced_complex_test.npz
```

### loss 配置

使用当前 v4 small polygon 候选配置：

```text
loss_type = weighted_mse_dice_area
defect_weight = 5
lambda_dice = 0.03
lambda_area = 0.04
lambda_tv = 0
area_loss_type = symmetric
physics_loss = off
L-BFGS = off
```

### seed 与 epoch

固定：

```text
seed = 42
epochs = 100
```

由于第 7.12 到第 7.17 的历史模型是在 `--seed` 加入前训练的，第 7.20 的公平对比应包含一个旧结构 seed=42 baseline。

建议第 7.20A 做两组：

1. `baseline` 结构 + seed=42 + 同一 loss 配置；
2. `calibrated_mu` 输出参数化 + 当前 decoder + seed=42 + 同一 loss 配置。

历史模型 `checkpoints/best_model_v4_w5_dice003_area004.pt` 只作为参考，不作为唯一对照。

### checkpoint 命名

建议：

```text
checkpoints/best_model_v4_baseline_w5_dice003_area004_seed42.pt
checkpoints/best_model_v4_calibrated_mu_w5_dice003_area004_seed42.pt
```

如进入第 7.20B，再使用新的 decoder 命名，例如：

```text
checkpoints/best_model_v4_calibrated_mu_decoder256_w5_dice003_area004_seed42.pt
```

### 结果文件命名

建议：

```text
results/loss_curves/loss_curve_v4_baseline_w5_dice003_area004_seed42.png
results/loss_curves/loss_curve_v4_calibrated_mu_w5_dice003_area004_seed42.png

results/previews/reconstruction_preview_v4_baseline_w5_dice003_area004_seed42.png
results/previews/reconstruction_preview_v4_calibrated_mu_w5_dice003_area004_seed42.png

results/metrics/evaluation_metrics_v4_baseline_w5_dice003_area004_seed42.csv
results/metrics/evaluation_metrics_v4_calibrated_mu_w5_dice003_area004_seed42.csv

results/summaries/v4_calibrated_mu_structure_summary.txt
```

如进入第 7.20B，再新增独立 summary，例如：

```text
results/summaries/v4_calibrated_mu_decoder_structure_summary.txt
```

### 对比对象

主要对比：

1. 新训练的 seed=42 baseline 结构；
2. 新训练的 seed=42 calibrated_mu 输出参数化结构；
3. 历史第 7.17 候选模型，仅作参考：
   `checkpoints/best_model_v4_w5_dice003_area004.pt`

### 重点指标

标准指标仍使用 threshold=500：

1. overall MSE / MAE / IoU / Dice / area_error / center_error；
2. polygon area_error；
3. small polygon pred_area=0 数量；
4. small polygon IoU / Dice；
5. medium polygon area_error；
6. multi_defect center_error；
7. pred_area > true_area 数量。

同时建议新增诊断统计，但不改变标准指标定义：

1. true defect pixels 上预测 μ_r 的 mean / median / p10 / p90；
2. true defect pixels 中 `μ_r < 100`、`μ_r < 300`、`μ_r < 500` 的比例；
3. 背景区域被预测为 `μ_r < 500` 的比例；
4. threshold=300 与 threshold=500 的 area_error 差距是否缩小。

## 5. 风险点

1. sigmoid 输出参数化可能带来梯度饱和，训练初期需要观察 loss 是否下降。
2. bounded μ 输出可能让背景更稳定，但也可能导致小缺陷边界过硬或过小。
3. 第 7.20A 不改 decoder，如果效果有限，不能直接说明结构方向失败，只能说明单独输出校准不足。
4. 第 7.20B 的 decoder 增强会增加参数量，可能改善表达能力，也可能增加过拟合风险。
5. 历史第 7.12 到第 7.17 模型没有固定 seed，不能和第 7.20A 的 seed=42 实验做完全严格的一对一比较。
6. 如果新结构只改善 threshold=300 下的后处理指标，而标准 threshold=500 没改善，则不能作为主方案。

## 6. 完成标准

第 7.20A 可以认为有效，需要至少满足：

1. small polygon `pred_area=0` 保持为 0/25 或不明显退化；
2. standard threshold=500 下的 overall IoU / Dice 不明显下降；
3. overall area_error 或 polygon area_error 明显低于 seed=42 baseline；
4. true defect pixels 的预测 μ_r 更接近低 μ 区间，`μ_r < 300` 的比例提高；
5. threshold=300 与 threshold=500 的 area_error 差距缩小，说明输出校准改善；
6. multi_defect center_error 不明显恶化。

如果只改善某一个局部指标，但明显损害 overall IoU / Dice 或 multi_defect center_error，则只记录为结构实验结果，不切换 baseline。

## 7. 暂不建议继续做的方向

1. 暂不继续大规模 loss 权重细扫。第 7.12 到第 7.17 已显示 loss 调参收益递减。
2. 暂不把 over-only area loss 作为主方案。它会让欠预测的小 polygon 缺少面积修正梯度，已导致 small polygon 漏检回升。
3. 暂不频繁用 test 集调参。后续结构实验仍应主要看 val 集，test 只做阶段性最终评估。
4. 暂不优先重写 BzEncoder。第一轮结构优化应先验证输出参数化和 decoder 校准。
5. 暂不把 threshold=300 后处理当作替代方案。它说明模型校准存在问题，但不能替代标准 threshold=500 的评价。
6. 暂不重新加入 physics_loss、L-BFGS、focal loss 或 oversampling，避免一次引入多个变量。
