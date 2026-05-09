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
