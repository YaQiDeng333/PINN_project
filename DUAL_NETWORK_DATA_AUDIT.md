# 双网络变分支线的数据结构核查

本文档记录 `feature/dual-network-variational` 支线在进入代码骨架前的数据结构核查结果。当前任务只做只读核查和支线文档整理，不修改 `README.md`、`train_pinn.py`、`data_generator_v2.py`、`MODEL_STRUCTURE_PLAN.md`，不训练，不提交 Git，不 push。

核查范围：

- 当前 worktree 实际存在的 `data_generator_v2.py`。
- 当前 worktree 实际存在的 `DUAL_NETWORK_IMPLEMENTATION_PLAN.md`。
- 当前 worktree 顶层文件和 `data/`、`datasets/`、`outputs/`、`results/`、`runs/`、`checkpoints/` 等数据 / 输出目录。
- 当前 worktree 中不存在 `train_pinn.py`；训练读取结构只能从 `origin/main:train_pinn.py` 只读核查，未将该文件检出到支线。

## 一、数据生成结构

### 1. x / y 网格范围和数量

`data_generator_v2.py` 的 `_generate_dataset` 使用规则二维网格：

```text
x = linspace(-15, 15, grid_size[1])
y = linspace(0, 10, grid_size[0])
X, Y = meshgrid(x, y)
```

默认参数为：

```text
grid_size = (100, 200)
```

因此默认：

- `x` 范围是 `[-15, 15]`，网格点数是 `200`。
- `y` 范围是 `[0, 10]`，网格点数是 `100`。
- `X`、`Y` 的形状是 `[100, 200]`，含义是物理二维区域内每个网格点的坐标，注释单位为 mm。

命令行入口也使用：

```text
--grid-y default 100
--grid-x default 200
```

### 2. bz_signal 的生成方式

每个样本先生成缺陷 mask，并把缺陷面积像素数、缺陷深度和 lift-off 放进解析近似信号公式：

```text
signal_amp = defect_area_pixels * sample_depth * 0.12
dist = sqrt((X - center_x)^2 + (sample_lift_off + (10 - Y))^2)
bz_signal = B0 + (X - center_x) / (dist^3 + 1e-6) * signal_amp
```

其中：

- `B0 = 1.5`。
- `defect_area_pixels = sum(mask)`。
- `sample_depth` 从 `depth` 范围采样，默认范围为 `(0.5, 2.0)`。
- `sample_lift_off` 从 `lift_off` 参数采样，默认值为 `2.0`。

这不是由完整磁静态 PDE 正问题求解得到的 Bz，而是与缺陷面积、深度、横向位置和 lift-off 相关的解析近似 / 构造式信号。

### 3. signals 是否来自 bz_signal[-1, :]

确认是。生成逻辑为：

```text
bz_at_liftoff = bz_signal[-1, :]
noise = normal(0, sample_noise_level, bz_at_liftoff.shape)
bz_noisy = bz_at_liftoff + noise
signals[idx] = bz_noisy
```

因此保存到 `.npz` 的 `signals` 是 `bz_signal` 最后一行沿 `x` 方向的 Bz 序列，并叠加高斯噪声。

### 4. lift_off 在 Bz 信号中的体现

`lift_off` 不作为单独的 `y` 坐标保存到 `signals` 中，而是进入距离项：

```text
sample_lift_off + (10 - Y)
```

当取 `bz_signal[-1, :]` 时，对应 `Y = 10`，距离项退化为：

```text
sample_lift_off
```

所以当前 `signals` 已经包含 lift-off 对解析 Bz 信号的影响。第一版支线实现可以把探头线放在计算域顶部 `y_s = y_max = 10.0`，不需要把 `PhiNet` 的求导点直接扩展到 `10.0 + lift_off`。若后续显式扩展空气域，再重新考虑域外 lift-off 探头线。

### 5. mu_map 的范围、尺度和形状

每个样本的 `mu_map` 初始化为背景磁导率：

```text
mu_bg = 1000.0
mu_map = ones(grid_size) * mu_bg
```

缺陷区域被设置为：

```text
mu_map[mask] = 1.0
```

因此当前 simple 数据中：

- 背景约为 `mu = 1000.0`。
- 缺陷约为 `mu = 1.0`。
- 单个 `mu_map` 形状为 `[grid_y, grid_x]`，默认 `[100, 200]`。
- 整体 `mu_maps` 形状为 `[num_samples, grid_y, grid_x]`，默认 `[N, 100, 200]`。

这与主线训练中的 `MU_SCALE = 1000.0` 约定兼容。

### 6. coords 的形状和含义

`data_generator_v2.py` 不保存名为 `coords` 的字段。它只保存一维 `x` 和 `y`，并在内部用 `np.meshgrid(x, y)` 构造 `X/Y`。

主线训练侧会从 `x/y` 重建坐标网格。以默认网格为例：

```text
coords shape = [grid_y * grid_x, 2] = [20000, 2]
coords[i] = (x_i, y_i) 或归一化后的 (x_i, y_i)
```

对支线来说，第一版应从 `.npz` 的 `x` 和 `y` 显式重建物理坐标 `coords_phys`，用于：

- `PhiNet(coords_phys) -> phi`；
- `MuNet(coords_phys) -> mu`；
- `Omega` 内部弱形式积分；
- 顶部探头线 `y_s = 10.0` 的 `Bz_pred = -phi_y(x, y_s)`。

如果后续使用归一化坐标，需要额外处理 `phi_x`、`phi_y` 的尺度换算，不能把归一化坐标导数直接当成物理导数。

### 7. 保存的数据字段

`_save_dataset` 保存字段为：

```text
signals
mu_maps
defect_types
metadata
x
y
```

其中 `metadata` 是结构化数组，字段包括：

```text
defect_type
center_x
center_y
width
height
radius
ellipse_a
ellipse_b
angle
triangle_vertices
area
depth
lift_off
noise_level
```

当前生成器会保存 `metadata`。但如果后续使用旧数据文件或外部数据文件，需要检查是否真的包含 `metadata`；如果没有，第一版支线只能从 `data_generator_v2.py` 默认参数和显式配置中恢复必要信息。

## 二、训练代码读取结构

### 1. 当前支线 worktree 中的 train_pinn.py 状态

当前 `feature/dual-network-variational` worktree 顶层不存在 `train_pinn.py`。因此：

- 不能从当前工作区本地文件直接确认主线训练读取结构。
- 以下内容来自只读检查 `origin/main:train_pinn.py`。
- 本次核查没有把 `train_pinn.py` 检出到支线，也没有修改主线代码。

### 2. Dataset / DataLoader 读取字段

`origin/main:train_pinn.py` 中的 `MFLDataset` 读取：

```text
signals
mu_maps
defect_types
metadata
metadata_keys, 如果存在
x
y
```

并派生：

```text
depths = metadata['depth']
lift_offs = metadata['lift_off']
```

读取后处理：

- `signals` 转为 `float32`，并按训练集均值 / 标准差标准化。
- `mu_maps` 转为 `float32` 后除以 `MU_SCALE = 1000.0`。
- `x`、`y` 转为 `float32`。

`__getitem__` 返回：

```text
signal:    [signal_length]
mu_target: [grid_y * grid_x]
idx:       sample index
```

默认 simple 数据下：

```text
signal shape    = [200]
mu_target shape = [100 * 200] = [20000]
```

### 3. signals、coords、mu_maps 进入模型的形状

主线 DataLoader batch 中：

```text
signals    -> [batch, signal_length]
mu_targets -> [batch, grid_y * grid_x]
```

坐标由 `build_coord_grid(x, y)` 构造：

```text
x_norm = x / max(abs(x.min), abs(x.max))
y_norm = 2 * (y - y.min) / (y.max - y.min) - 1
coords = stack(meshgrid(x_norm, y_norm)).reshape(-1, 2)
```

默认 simple 数据下：

```text
coords -> [20000, 2]
```

模型 forward 时，如果 `coords` 是 `[N_points, 2]`，会扩展为：

```text
coords -> [batch, N_points, 2]
```

预测输出：

```text
pred -> [batch, N_points]
pred_map -> [batch, grid_y, grid_x]
```

### 4. 当前主线模型输入输出

主线模型是：

```text
Bz signal -> BzEncoder -> latent
coords -> Fourier feature
concat(latent, coord_features) -> MLP -> mu_norm(x, y)
```

输入：

```text
signals: [batch, signal_length]
coords:  [N_points, 2] 或 [batch, N_points, 2]
```

输出：

```text
mu_norm: [batch, N_points]
```

预测完整图时再 reshape 并乘以：

```text
MU_SCALE = 1000.0
```

得到真实尺度的 `mu_r`。

### 5. 当前主线是否是监督反演

确认是。当前主线本质是：

```text
signals / coords -> mu_map
```

的单网络监督反演。

训练目标直接对齐 `mu_maps / MU_SCALE`，支持 MSE、weighted MSE、soft Dice、area loss、TV 等项。即使开启主线的 `physics_loss`，该项也是从预测 `mu_map` 估计简化 Bz 信号的辅助约束，不是求解真实 PDE 后得到训练标签。

### 6. 当前主线是否使用真实 PDE 求解

当前主线没有使用真实 PDE / FEM / COMSOL 正问题求解。训练标签来自 `data_generator_v2.py` 构造的 `mu_maps`，Bz 信号来自数据生成器的解析近似公式。

这意味着主线和支线都必须注意：当前数据足以验证工程闭环和梯度路径，但不能直接支持“严格物理正问题一致”的论文表述。

## 三、对支线实现的影响

### 1. 第一版是否可以把 signals 当作 Bz_meas

可以。`signals` 明确保存的是沿 `x` 方向的一维 Bz 序列，且来自 `bz_signal[-1, :]` 加噪后结果。第一版支线可将：

```text
Bz_meas = signals[sample_idx]
```

用于：

```text
L_data = mean((-phi_y(x, y_s) - Bz_meas)^2)
```

注意事项：

- 若使用主线式标准化信号，则 `Bz_pred` 也要使用同一 `signal_mean / signal_std` 标准化。
- 若第一版只做单样本闭环，建议先使用原始 `signals`，减少标准化和物理量纲混淆。

### 2. 第一版是否可以取 y_s = y_max = 10.0

可以，作为第一版实现假设是合理的。

依据：

- 数据网格 `y` 范围为 `[0, 10]`。
- `signals` 来自 `bz_signal[-1, :]`，即 `Y = 10` 的顶部网格行。
- `lift_off` 已经体现在解析距离项 `sample_lift_off + (10 - Y)` 中。

因此第一版可定义：

```text
y_s = y_max = 10.0
```

并明确限制：这不是把探头真实空间位置建模为 `10.0 + lift_off`，而是在当前生成器定义下复用顶部行的等效测量信号。只有后续显式扩展空气域时，才考虑域外 lift-off 探头线。

### 3. 第一版是否可以用 coords 构造 Omega 内部积分点

可以，但需要从 `x/y` 重建，而不是从数据文件读取 `coords` 字段。

第一版建议：

- 用 `np.meshgrid(x, y)` 构造物理坐标。
- 将坐标 reshape 为 `[grid_y * grid_x, 2]`。
- 使用物理坐标计算 `dx`、`dy` 和弱形式积分权重。
- 如果为了网络稳定使用归一化坐标，需要保留物理坐标用于导数尺度换算。

### 4. 第一版是否可以用 mu_maps 作为验证标签或可选监督先验

可以。

建议定位：

- `mu_maps` 首先作为验证标签 / 诊断指标，用于观察 `mu-Net` 是否反演到正确区域和尺度。
- 可选地用于 `mu_0` warm start 或轻量 `L_prior`，但不应成为固定 `phi` 后更新 `mu` 的核心目标。
- 若大量依赖 `mu_maps` 监督，支线会退化成 main 的监督反演，不再是弱形式材料更新方案。

### 5. 是否需要重新生成包含 metadata 的数据

当前 `data_generator_v2.py` 会保存 `metadata`，理论上重新生成 simple 数据时不需要额外改生成器。

但当前 worktree 中没有 `data/` 或 `.npz` 文件，无法验证实际本地数据文件是否包含 `metadata`。下一步如果要跑最小实验，需要先准备或生成一个 simple `.npz`，并检查：

```text
signals
mu_maps
metadata
x
y
```

是否都存在。

### 6. 是否需要显式保存 lift_off、x_grid、y_grid、y_s 等信息

建议第一版支线代码显式保存或记录以下信息：

- `x`、`y`：已经在 `.npz` 中保存，应直接读取。
- `x_grid`、`y_grid` 或 `coords_phys`：可由 `x/y` 重建，但支线结果日志中应记录 shape 和范围。
- `lift_off`：当前在 `metadata['lift_off']` 中保存；若数据没有 metadata，应在配置中显式给出默认值。
- `y_s`：当前不是数据字段，应在支线配置 / 日志中显式保存为 `10.0`。
- `dx`、`dy`：弱形式积分需要，建议由 `x/y` 计算并记录。
- `signal_mean`、`signal_std`：如果使用标准化信号，必须保存到 checkpoint 或实验日志。

## 四、风险和限制

### 1. signals 的物理一致性风险

当前 `signals` 可能是解析近似生成的漏磁信号，不一定来自严格 PDE 正问题。具体来说，Bz 来自：

```text
B0 + (X - center_x) / (dist^3 + 1e-6) * signal_amp
```

而不是由给定 `mu_map` 解 `div(mu grad phi)=0` 后得到的边界观测。

### 2. 支线第一版的结论边界

因此，支线第一版只能作为：

- 数据读取闭环验证；
- `PhiNet` 数据拟合闭环验证；
- 固定 `phi` 后 `MuNet` 是否能通过弱形式残差获得梯度的验证；
- `phi/mu` 交替流程是否能稳定运行的工程验证。

不能直接声称：

- 当前数据严格满足支线 PDE 假设；
- 当前弱形式残差与生成器 Bz 信号完全一致；
- 第一版实验已经证明真实物理反演有效。

### 3. 严谨数据需求

如果后续要做严谨物理结论，需要补充由真实正问题产生的数据，例如：

- FEM / COMSOL 生成的 Bz 数据；
- 与 `mu_map` 一致的 PDE 数值解；
- 明确空气域、材料域、边界条件和探头 lift-off 的计算域定义；
- 已知 Neumann 或 Dirichlet 边界条件；
- 多 lift-off 或多测线观测，以缓解欠定性。

### 4. metadata 缺失风险

当前生成器会保存 `metadata`，但如果实际数据文件来自旧版本或外部来源，可能没有 `metadata` 字段。

如果数据文件中没有 `metadata`：

- 第一版只能从 `data_generator_v2.py` 中读取默认参数；
- `lift_off`、`noise_level`、`depth` 等需要硬编码到支线配置或设为未知；
- `y_s = 10.0` 仍可由 `x/y` 网格和 `signals = bz_signal[-1, :]` 的生成器假设确定；
- 不能进行依赖逐样本 `lift_off` 的严谨诊断。

### 5. 当前目录的数据文件状态

当前 worktree 顶层没有以下目录：

```text
data/
datasets/
outputs/
results/
runs/
checkpoints/
```

也没有发现 `.npz`、`.npy`、`.pt`、`.pth`、`.csv` 等数据或训练输出文件。下一步最小实验前需要准备数据文件，但本阶段不生成数据、不训练。

## 五、结论

### 1. 当前数据结构是否足够支持最小可运行支线实验

从 `data_generator_v2.py` 的结构看，当前数据格式足够支持最小可运行支线实验：

- `signals` 可作为 `Bz_meas`。
- `x/y` 可重建物理坐标网格和 `Omega` 积分点。
- `y_s` 第一版可定义为 `y_max = 10.0`。
- `mu_maps` 可作为验证标签、诊断指标或可选 warm start。
- `metadata` 可提供 `lift_off`、`depth`、`noise_level` 等诊断信息。

但当前 worktree 没有实际 `.npz` 数据文件，所以代码实现前还需要准备或生成一个 simple 数据文件用于闭环验证。

### 2. 还缺哪些字段或配置

第一版实现前建议显式补充到支线配置或实验日志：

- `y_s = 10.0`。
- `dx`、`dy` 和弱形式积分权重定义。
- 是否使用原始 `signals` 还是标准化 `signals`。
- `signal_mean`、`signal_std`，如果使用标准化。
- `mu` 使用真实尺度还是归一化尺度。
- `phi` 边界条件。
- 测试函数 `v_q` 的具体形式和归一化方式。
- 如果实际数据缺少 `metadata`，则需要显式配置 `lift_off`、`depth`、`noise_level` 的默认值或标记为未知。

### 3. 下一步是否可以开始创建支线代码骨架

可以开始创建支线代码骨架，但应限定为最小闭环骨架，不直接进入大规模训练：

- `dual_network_models.py`：定义 `PhiNet` 和 `MuNet`。
- `dual_network_losses.py`：定义 `energy_loss`、`data_loss`、`weak_form_loss`、`tv_loss` 和积分工具。
- `train_dual_variational.py`：只支持单样本或极小数据闭环，显式读取 `signals/x/y/mu_maps/metadata`。
- `evaluate_dual_variational.py`：只输出 `phi`、`mu`、`Bz_pred vs Bz_meas` 和弱残差诊断。

进入代码前仍需确认：

- 是否先生成一个 simple `.npz` 数据文件；
- 是否使用原始信号尺度；
- 第一版边界条件如何设置；
- 第一版局部测试函数的窗口大小和积分权重。
