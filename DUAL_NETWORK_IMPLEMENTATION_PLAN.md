# 变分场重构与弱形式材料更新双网络 PINN 的实现方案

本文档是 `feature/dual-network-variational` 支线的实现前技术设计。当前阶段只做方案设计，不写训练代码，不修改 main 主线脚本，不提交 Git，不 push。

读取说明：当前支线 worktree 的 HEAD 中没有 `train_pinn.py` 和 `evaluate_pinn.py`，但它们存在于 `origin/main`。本方案对这两个文件的梳理来自只读查看 `origin/main:train_pinn.py` 和 `origin/main:evaluate_pinn.py`；不会把它们检出或同步到支线。

## 一、当前主线代码结构梳理

### data_generator_v2.py

当前支线 worktree 中的 `data_generator_v2.py` 提供 simple 缺陷数据生成流程：

- 空间网格：`x = linspace(-15, 15, grid_x)`，`y = linspace(0, 10, grid_y)`。
- 默认网格：`grid_size=(100, 200)`，因此 `mu_maps` 形状是 `[N, 100, 200]`，`signals` 形状是 `[N, 200]`。
- `mu_maps`：背景 `mu_bg = 1000.0`，缺陷区域 `mu = 1.0`。
- `signals`：一维 lift-off 探头线 Bz 数据，保存字段名为 `signals`。
- `metadata`：包含 `defect_type`、缺陷几何参数、`depth`、`lift_off`、`noise_level` 等。
- 保存格式：`.npz` 中包含 `signals`、`mu_maps`、`defect_types`、`metadata`、`x`、`y`。

从 `origin/main:data_generator_v2.py` 只读检查可见，main 版本已经扩展了 v3/v4 complex 数据、`metadata_keys`、`mask_pixels`、`signal_snr`、`area_bin` 等字段。支线第一版实现应优先兼容 simple 数据字段，同时在读取器里容忍 main 版本新增字段。

关键判断：

- 当前数据集中已经包含 Bz 探头线数据，字段是 `signals`。
- `signals` 是沿 `x` 方向的一维 Bz 序列，对应 lift-off 探头线。
- `mu_maps` 是二维相对磁导率标签，shape 为 `[sample, y, x]`。
- `x/y` 是物理坐标，单位按现有注释为 mm。

### train_pinn.py

`train_pinn.py` 当前不在支线 HEAD 中；以下来自 `origin/main` 只读结构。

main 训练脚本职责：

- `MFLDataset` 读取 `.npz`，将 `signals` 标准化，将 `mu_maps` 除以 `MU_SCALE = 1000.0` 后展平成监督目标。
- `build_coord_grid(x, y)` 生成归一化坐标网格。
- `feature_mapping(coords)` 对坐标做 Fourier feature 编码。
- `BzEncoder` 用一维卷积把 Bz signal 编码为 latent vector。
- `PINN` 将 `Bz latent + coord features` 拼接后输入 MLP，输出归一化 `mu(x, y)`。
- `run_epoch` 支持 `mse`、`weighted_mse`、`weighted_mse_dice`、`weighted_mse_dice_area`、TV loss 和简化 `physics_loss`。
- `train_adam_tv` 是主训练入口，直接以监督 `mu_maps` 为目标训练 `mu`。

关键判断：

- 当前主线网络输入是 `Bz signal + 空间坐标 (x, y)`。
- 当前主线网络输出是归一化后的 `mu map`，预测后乘以 `MU_SCALE` 回到真实 `mu_r`。
- 当前训练目标本质是直接监督预测 `mu`，不是通过求解 `phi` 场或弱形式材料更新得到 `mu`。
- 当前 `physics_loss` 是从预测 `mu_map` 估计简化 Bz 信号的辅助项，不是本支线所需的 `div(mu grad phi)=0` 弱形式。

可复用部分：

- `.npz` 数据读取逻辑和字段约定。
- `MU_SCALE = 1000.0`、`MASK_THRESHOLD = 500.0` 这类尺度和评估阈值约定。
- `build_coord_grid` 的坐标范围处理思路，但支线需要保留可微坐标用于求导。
- `tv_loss` 的基本形式可作为参考。
- seed、路径安全、结果命名等工程习惯。

不应直接复用为核心的部分：

- `BzEncoder + PINN` 单网络结构。
- `run_epoch` 中直接监督 `mu_maps` 的训练目标。
- main 的 `physics_loss`，因为它是简化 forward Bz 重建，不是弱形式材料更新。
- main 的训练入口 `train_pinn.py`。支线应新建独立入口，避免污染主线。

### evaluate_pinn.py

`evaluate_pinn.py` 当前不在支线 HEAD 中；以下来自 `origin/main` 只读结构。

main 评估脚本职责：

- 从 `train_pinn.py` 导入 `MFLDataset`、`PINN`、`MU_SCALE`、`build_coord_grid`。
- 加载 checkpoint，使用 main 单网络预测 `mu_map`。
- 使用 `MASK_THRESHOLD = 500.0` 将 `mu < 500` 判为缺陷区域。
- 输出 MSE、MAE、IoU、Dice、area_error、center_error。
- 保存 metrics CSV/TXT 和预测对比图。

可复用部分：

- `compute_sample_metrics` 的指标定义。
- mask threshold 约定。
- 可视化输出布局思路。

不应直接复用为核心的部分：

- checkpoint 加载方式绑定 main 的 `PINN`。
- `predict_batch_maps` 假设模型输入为 `signals + coords` 且输出 `mu`。
- 支线需要同时评估 `phi`、`mu`、探头线 Bz 拟合曲线和弱形式残差，因此应新建 `evaluate_dual_variational.py`。

## 二、支线新代码建议结构

不要直接改 `train_pinn.py`。建议新增独立支线文件：

### train_dual_variational.py

职责：

- 支线训练入口。
- 读取 `.npz` 中的 `signals`、`mu_maps`、`x`、`y`、metadata。
- 构建可微坐标张量和探头线坐标。
- 初始化 `PhiNet`、`MuNet`、两个 optimizer。
- 实现 `phi_step` / `mu_step` 外层交替循环。
- 保存支线 checkpoint、loss 日志和最小可视化结果。

### dual_network_models.py

职责：

- 定义 `PhiNet`。
- 定义 `MuNet`。
- 定义可选坐标编码，例如 Fourier features 或 sinusoidal encoding。
- 定义 `mu` 输出参数化，例如正值约束或归一化范围映射。

### dual_network_losses.py

职责：

- 实现 `energy_loss`、`data_loss`、`weak_form_loss`、`tv_loss` 等支线损失。
- 封装弱形式积分工具，包括积分点、积分权重和测试函数求导。
- 生成局部 compact-support test functions。
- 计算 `grad(phi)`。
- 封装边界条件 mask、Dirichlet hard constraint 或后续 Neumann 边界积分。

### evaluate_dual_variational.py

职责：

- 加载支线 checkpoint。
- 输出 `phi` 场、`mu` 场、`Bz_pred = -phi_y` 与 `Bz_meas` 曲线。
- 输出弱形式残差统计。
- 可选复用 main 的 mask 指标定义评估 `mu`，但不改变 main 评估脚本。

## 三、核心模块设计

### 1. PhiNet

目标：

```text
input:  coords = (x, y)
output: phi(x, y)
```

设计要点：

- 第一版使用坐标 MLP。
- 输入坐标需要 `requires_grad=True`，以便自动微分得到 `phi_x`、`phi_y`。
- 可选使用 Fourier features，但要确保对原始物理坐标或归一化坐标的导数尺度定义清楚。
- 输出为标量势函数，不输出 `mu`。

### 2. MuNet

目标：

```text
input:  coords = (x, y)
output: mu(x, y)
```

设计要点：

- 第一版使用坐标 MLP。
- 输出应保持正值，例如 `mu = mu_min + softplus(raw_mu)`。
- 也可以输出归一化 `mu_norm`，再映射回真实 `mu_r`，但必须统一能量项和弱形式中的尺度。
- 第一版可从背景常数 `mu_bg` 初始化，或使用 `mu_maps` 做少量 warm start；`mu_maps` 不作为固定 `phi` 后的核心训练目标。

### 3. energy_loss

定义：

```text
L_energy = mean(0.5 * mu * (phi_x^2 + phi_y^2))
```

用途：

- 在固定 `mu_k` 的 `phi_step` 中使用。
- 约束 `phi` 场在当前材料分布下形成低能量解。
- 若使用归一化坐标，需要处理导数尺度，否则 `phi_x`、`phi_y` 的物理量纲会被改变。

### 4. data_loss

定义：

```text
L_data = mean((-phi_y(x, y_s) - Bz_meas)^2)
```

用途：

- 只用于 `phi_step`。
- `y_s` 是探头线位置，`Bz_meas` 来自 `.npz` 的 `signals` 字段。
- 如果训练读取的是标准化后的 `signals`，则 `Bz_pred` 也必须做同样标准化；否则使用原始 `signals`。

第一版探头线定义：

- 计算域沿用 `data_generator_v2.py` 中的二维网格：`x in [-15, 15]`，`y in [0, 10]`。
- 探头数据 `signals` 来自 `bz_signal[-1, :]`，因此第一版把探头线定义为 `y_s = y_max = 10.0`。
- 当前 `signals` 已经包含 `lift_off` 对解析 Bz 信号的影响；第一版不把 `PhiNet` 的求导点扩展到 `y = 10.0 + lift_off`。
- 只有在后续显式扩展空气域时，才考虑把探头线设置到域外 lift-off 位置。

禁止用法：

- 固定 `phi` 后，不用该数据损失更新 `MuNet`，因为此时 `L_data` 不含 `mu`。

### 5. weak_form_loss

定义：

```text
L_weak = mean_q | integral_Omega mu * grad(phi_fixed) dot grad(v_q) dOmega |^2
```

来源：

```text
div(mu grad phi) = 0
```

第一版假设：

- 无源区域。
- Dirichlet 或 hard Dirichlet 边界。
- 测试函数 `v_q` 在 Dirichlet 边界取零。
- 弱形式右端 `F(v_q)=0`。

测试函数：

- 第一版使用局部 compact-support test functions。
- 可在低分辨率窗口上构造三角窗、余弦窗或平滑 bump function。
- 后续再考虑随机平滑测试函数或 FEM 类基函数。

实现要点：

- `phi_fixed` 和 `grad(phi_fixed)` 必须 detach。
- 只让 `mu` 对 `L_weak` 接收梯度。
- 积分需要乘以网格 cell area 或明确采用 mean 近似。

### 6. tv_loss

定义：

```text
TV(mu) = mean(|mu[:, 1:] - mu[:, :-1]|) + mean(|mu[1:, :] - mu[:-1, :]|)
```

用途：

- 抑制 `mu` 场毛刺。
- 第一版只作为 `mu_step` 正则。

注意：

- TV 权重过大可能把缺陷抹平。
- TV 权重过小可能导致 `mu` 利用弱残差噪声产生非物理振荡。

### 7. alternating_train_loop

流程：

```text
for outer_iter in range(K):
    freeze(MuNet)
    unfreeze(PhiNet)
    optimize L_phi = L_energy + lambda_bc L_bc + lambda_data L_data

    phi_fixed = detach(PhiNet(coords))
    grad_phi_fixed = detach(grad(phi_fixed, coords))

    freeze(PhiNet)
    unfreeze(MuNet)
    optimize L_mu = L_weak + beta TV(mu) + gamma L_prior
```

记录项：

- `L_energy`
- `L_bc`
- `L_data`
- `L_weak`
- `TV(mu)`
- `L_prior`
- Bz 拟合误差
- `mu_maps` 诊断指标

## 四、目前最关键的未定参数

实现前必须确认：

- `Omega` 的坐标范围：当前 simple 数据是 `x in [-15, 15]`，`y in [0, 10]`。
- 探头线 `y_s` 的位置：第一版固定为 `y_s = y_max = 10.0`，因为 `signals` 来自 `bz_signal[-1, :]`；后续若扩展空气域，再重新定义域外 lift-off 探头线。
- `Bz_meas` 字段名：当前 `.npz` 中为 `signals`。
- `signals` 使用原始值还是标准化值：若标准化，需要保存并复用 `signal_mean/signal_std`。
- `mu` 的归一化尺度：main 使用 `MU_SCALE = 1000.0`，支线能量项和弱形式中必须统一使用真实 `mu` 或归一化 `mu`。
- `phi` 的边界条件：Dirichlet 值、hard constraint 形式或后续 Neumann 磁通边界。
- 测试函数 `v_q` 的形式：局部窗口、随机平滑函数还是 FEM 类基函数。
- 积分点采样方式：全网格、低分辨率网格、随机点还是窗口中心采样。
- 积分权重：是否乘以 `dx * dy`，以及归一化方式。
- 是否需要 hard constraint 边界条件。
- 输出缺陷区域时使用的 `mu` 阈值：main 评估使用 `mu < 500`。
- `mu_bg` 和缺陷 `mu` 的范围：当前 simple 数据为背景约 `1000`、缺陷约 `1`。
- `L_prior` 的定义：背景先验、正值先验、范围先验、稀疏先验或 warm start。

## 五、风险分析

### weak_form_loss 退化

如果 `grad(phi_fixed)` 在大区域内接近零，或者测试函数族覆盖不足，`L_weak` 可能无法给 `mu` 提供有效梯度。此时 `mu-Net` 可能只受 TV 和 prior 控制，学到平滑但无意义的材料场。

### phi-Net 只拟合探头线

`L_data` 只约束探头线上的 `-phi_y`。如果 `L_energy`、`L_bc` 太弱，`phi-Net` 可能只在探头线附近拟合 Bz，而全域 `phi` 场不满足合理物理结构，导致后续 `mu_step` 跟随错误场。

### mu-Net 产生平滑但错误分布

弱形式约束在单条探头信号下可能欠定。`mu-Net` 可能产生低残差、低 TV 的平滑分布，但缺陷位置和形状错误。因此必须用 `mu_maps` 做诊断，而不是只看 `L_weak`。

### TV 权重风险

- TV 过大：缺陷区域被抹平，`mu` 接近背景或低频场。
- TV 过小：`mu` 可能出现局部振荡、棋盘纹或利用测试函数盲区的非物理结构。

### 当前数据支撑风险

当前数据生成器的 Bz 信号是解析近似公式生成，不一定严格满足后续假设的 PDE 弱形式。尤其 simple 版本中 `mu_map` 与 `signals` 的关系是构造性的，不是完整磁静态数值解。因此第一版实验只能验证算法闭环和梯度路径，不能直接声称物理反演有效。

### phi/mu 补偿性漂移

`phi_step` 可能拟合探头信号但偏离真实全域场，`mu_step` 又跟随错误 `phi` 产生补偿性材料分布。表现为总 loss 下降，但 `mu_maps` 诊断指标变差。

## 六、最小可运行实验设计

目标：只验证训练闭环，不追求指标。

实验设置：

- 只选一个 simple 缺陷样本。
- 使用固定规则网格，例如当前 `100 x 200` 或更低分辨率子网格。
- 使用 `.npz` 中的 `signals[0]` 作为 `Bz_meas`。
- 使用 `.npz` 中的 `x/y` 构造坐标。
- `mu_0` 初始化为背景常数场。

步骤：

1. 构建 `PhiNet(coords) -> phi`。
2. 固定 `mu_0`，先训练 `PhiNet`，最小化 `L_phi`。
3. 输出 `phi` 场和 `Bz_pred = -phi_y` 与 `Bz_meas` 的曲线。
4. 固定训练后的 `phi` 和 `grad(phi)`。
5. 构建局部 compact-support test functions。
6. 训练 `MuNet`，最小化 `L_weak + beta TV(mu) + gamma L_prior`。
7. 输出 `mu` 场、弱残差曲线和 TV 曲线。

最小输出：

- `phi` 场图。
- `mu` 场图。
- `Bz_pred` vs `Bz_meas` 曲线。
- `L_phi`、`L_weak`、`TV(mu)` 日志。

不要求：

- 不要求 test 指标。
- 不要求大规模训练。
- 不要求超过 main baseline。
- 不要求保存正式 checkpoint。

## 七、当前不做的事情

当前阶段明确不做：

- 不修改主线 `train_pinn.py`。
- 不修改主线 `evaluate_pinn.py`。
- 不修改 `data_generator_v2.py`。
- 不修改 `README.md`。
- 不替换当前主线模型。
- 不进行大规模训练。
- 不生成正式 checkpoint。
- 不声称该支线已经优于主线。
- 不同步进 main。
- 不提交 Git。
