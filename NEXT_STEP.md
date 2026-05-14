NEXT_STEP

## 当前最新状态（以此为准）

第 7.23 步：`calibrated_mu` adaptive threshold calibration 已完成。

本轮是 evaluation-level adaptive threshold calibration，不重新训练、不修改模型结构、不修改 `CURRENT_BASELINE`。规则只使用 default threshold=500 下的 predicted area 分段选择 threshold，不使用 true area。

对比方法：

* default threshold=500；
* global calibrated threshold：standard=400，enhanced=350；
* adaptive threshold：根据 default `pred_area` 分段选择 threshold。

validation set 选出的 adaptive rule：

* standard：A=9.654345，B=12.387713，T_small=450，T_medium=350，T_large=350；
* enhanced：A=9.897988，B=15.232851，T_small=350，T_medium=350，T_large=300。

关键结论：

* standard adaptive：test area_error 从 default 0.829989 降到 0.474476，低于 global 0.513358；IoU/Dice 为 0.347960 / 0.485609；
* enhanced adaptive：test area_error 从 default 0.953397 降到 0.360101，低于 global 0.416337；IoU/Dice 为 0.350185 / 0.490040；
* adaptive 仍能明显降低 area_error，但 IoU / Dice 比 global threshold 更低；
* standard adaptive 比 standard global 更少伤害 small polygon：small pred_area=0 从 5.333 降到 3.000，small IoU=0 从 14.000 降到 12.667；
* enhanced adaptive 对 small polygon 的保护与 enhanced global 基本相同：small IoU=0 = 9.667，small pred_area=0 = 0.333；
* 当前不切换 `CURRENT_BASELINE`。

## 当前下一步

第 7.23 不用于更新 baseline，只用于评估层面的 adaptive threshold calibration 诊断。

adaptive threshold 作为 evaluation-level calibration 记录；是否作为后续候选由主线对话决定。

---

## 当前最新状态（以此为准）

第 7.20A 步：`calibrated_mu` 输出 μ 参数化校准实验已完成。

本轮已经在 `seed=42` 下完成 baseline 与 `calibrated_mu` A/B 对比。`calibrated_mu` 保持 BzEncoder 和 decoder 主体不变，只改变输出 μ 参数化，将 defect probability 映射到 `mu_norm ∈ [0.001, 1.0]`。

关键结论：

* `calibrated_mu` 相比 baseline seed=42 改善了 MSE、IoU、Dice、center_error、polygon area_error、small polygon IoU / Dice 和 multi_defect center_error；
* 缺陷区预测 μ_r 均值从约 399 降到约 361，中位数从约 295 降到约 262，说明输出校准方向有效；
* area_error 只从 0.6404435 降到 0.6401099，改善很小；
* `pred_area > true_area` 数量从 174 / 200 增加到 182 / 200；
* small polygon `pred_area=0` 保持 0 / 25；
* 当前不切换 `CURRENT_BASELINE`。

## 当前下一步建议

进入第 7.20B：decoder 轻量增强 A/B 实验。

建议保持：

* dataset = `v4_balanced_complex`
* seed = 42
* loss_type = `weighted_mse_dice_area`
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.04
* lambda_tv = 0
* area_loss_type = `symmetric`
* 不启用 physics_loss
* 不启用 L-BFGS
* 不切换 CURRENT_BASELINE

第 7.20B 只应在第 7.20A 的基础上测试轻量 decoder 增强，不要同时加入新的 loss、后处理或数据增强，避免混淆变量。

---

## 当前最新下一步（以此为准）

第 7.20A 步：输出 μ 参数化校准实验。

第 7.19 步已经完成方案设计，详见 `MODEL_STRUCTURE_PLAN.md`。核心判断是：第 7.12-7.18 的实验已经说明 weighted MSE、soft Dice Loss、area-aware loss 和后处理阈值可以缓解 small polygon 漏检和面积误差，但缺陷区域预测 μ 值仍偏软，常停留在 `μ_r≈200-400`，而不是接近真实缺陷 `μ_r≈1`。当前输出层实际是 `Linear + Softplus`，有下界但无上界；缺陷端要逼近 `mu_norm≈0.001` 时需要很负的 pre-activation，可能导致输出偏软。因此下一步应优先单独验证输出参数化，而不是继续扩大 loss 调参。

第 7.20A 推荐实验：

* 数据集：`data/training_data_v4_balanced_complex_train.npz`、`data/training_data_v4_balanced_complex_val.npz`、`data/training_data_v4_balanced_complex_test.npz`
* 固定 seed：`42`
* loss 配置：`weighted_mse_dice_area`
* `defect_weight = 5`
* `lambda_dice = 0.03`
* `lambda_area = 0.04`
* `lambda_tv = 0`
* `area_loss_type = symmetric`
* 不启用 physics_loss
* 不启用 L-BFGS
* 不切换 CURRENT_BASELINE

第 7.20A 最小实现方向：

1. 在 `train_pinn.py` 中新增模型结构开关，例如 `--model-variant baseline / calibrated_mu`；
2. 默认仍为 `baseline`，保证旧训练流程不受影响；
3. 新增 calibrated μ 输出参数化，让 decoder 先预测 defect probability，再映射到物理合理的归一化 μ 范围 `[0.001, 1.0]`；
4. 保持当前 decoder 结构不变，即 `128 / 128 / 64 + Tanh`；
5. 如需评估新 checkpoint，仅对 `evaluate_pinn.py` 做兼容模型变体加载的最小修改，不改变 MSE、MAE、IoU、Dice、area_error、center_error 定义。

建议第 7.20A 做公平 A/B：

1. 旧结构 + seed=42 + 同一 loss 配置；
2. calibrated_mu 输出参数化 + 当前 decoder + seed=42 + 同一 loss 配置。

第 7.20B 暂不立即执行。只有当第 7.20A 有效或部分有效后，再考虑增强 decoder，例如 `256 / 256 / 128 / 64 + SiLU`，避免一次性同时改变输出参数化和 decoder 容量。

重点观察：

* standard threshold=500 下的 overall IoU / Dice / area_error；
* polygon area_error；
* small polygon `pred_area=0` 数量；
* true defect pixels 上预测 μ_r 是否更接近低 μ 区间；
* threshold=300 与 threshold=500 的 area_error 差距是否缩小；
* multi_defect center_error 是否恶化。

---

当前状态（最新）

第 7.9 步：v4 balanced complex 正式规模数据集生成已完成。

当前推荐 v3_complex 模型

checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt

v4_balanced_complex 数据集

data/training_data_v4_balanced_complex_train.npz

data/training_data_v4_balanced_complex_val.npz

data/training_data_v4_balanced_complex_test.npz

样本数量

train = 1000

val = 200

test = 200

生成配置

seed = 7904

polygon mask_pixels >= 30

polygon signal_snr >= 5

area_bin 阈值：

small: mask_pixels < 120

medium: 120 <= mask_pixels < 500

large: mask_pixels >= 500

multi_defect 2 缺陷 / 3 缺陷约为 40% / 60%

正式检查摘要

results/summaries/v4_balanced_complex_dataset_summary.txt

关键检查结论

metadata_keys 与 metadata 字段一致。

signals 和 mu_maps 无 NaN / Inf。

每个样本缺陷 mask 非空。

train polygon area_bin 分布：

small = 124

medium = 103

large = 73

train multi_defect num_defects 分布：

2 defects = 120

3 defects = 180

当前下一步

可以进入 v4_balanced_complex 模型训练。

建议训练新的独立 v4 baseline，不覆盖当前 v3_complex 推荐模型。

执行约束

不要修改 data_generator_v2.py，除非后续 review 明确要求。

不要修改 evaluate_pinn.py 的评价指标定义。

不要启用 physics_loss。

不要启用 L-BFGS。

不要覆盖当前推荐 v3_complex 模型：

checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt

---

## 当前最新状态：第 7.10 步已完成

v4_balanced_complex baseline 已训练完成：

* 模型：checkpoints/best_model_v4_balanced_complex_tv.pt
* lambda_tv = 2e-6
* epoch = 100
* physics_loss：未启用
* L-BFGS：未启用
* 评估文件：results/metrics/evaluation_metrics_v4_balanced_complex_tv.csv
* 诊断摘要：results/summaries/v4_balanced_complex_diagnosis_summary.txt

当前判断：v4 baseline 没有明显优于当前 v3_complex 推荐模型，因此 CURRENT_BASELINE.md 暂不更新。

当前推荐 v3_complex 模型仍为：

checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt

## 下一步建议

第 7.11 步：v4_balanced_complex 专属 lambda_tv 扫描。

建议候选值：0、1e-6、2e-6、5e-6、1e-5。主要基于 val 集选择，test 集只用于最终确认。暂时不要加入 physics_loss、L-BFGS 或模型结构改动。

---

## 当前最新状态：第 7.11 步已完成

v4_balanced_complex 专属 lambda_tv 扫描已完成：

* 候选值：0、5e-7、1e-6、2e-6、5e-6、1e-5
* 每组训练：50 epoch
* physics_loss：未启用
* L-BFGS：未启用
* 模型结构：未修改
* 评价指标定义：未修改

按 val_iou、val_dice、val_mae、val_area_error、val_center_error 综合排序，本轮 v4 推荐候选为：

checkpoints/best_model_v4_balanced_complex_tv_sweep_0.pt

对应 lambda_tv = 0。

该候选 test 指标：

* MSE = 2.41644578e+04
* MAE = 5.09550103e+01
* IoU = 2.73743067e-01
* Dice = 3.87241381e-01
* area_error = 4.90251054e-01
* center_error = 1.38652205e+00

关键结论：

* small polygon 没有改善，IoU = 0，Dice = 0；
* multi_defect center_error 相比第 7.10 v4 baseline 略有改善，但仍不优于当前 v3_complex 推荐模型；
* v4 sweep 候选没有明显超过 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt；
* CURRENT_BASELINE.md 不切换。

## 下一步建议

第 7.12 步：进入模型结构或训练策略优化方案设计。

建议优先讨论：

1. small polygon 漏检的针对性训练策略；
2. mask-aware loss / focal 类 loss 是否适合加入；
3. BzEncoder 或空间解码结构是否需要增强；
4. 是否需要为 small polygon 或 multi_defect 做采样权重，而不是继续扩大 lambda_tv 扫描。

---

## 当前最新状态：第 7.12A 步已完成

已完成 small polygon 漏检专项的第一轮训练策略实验：defect-weighted MSE Loss。

本轮只修改 `train_pinn.py` 的 loss 选择逻辑，未修改数据生成器、评价指标定义或模型结构。

训练配置：

* 数据集：v4_balanced_complex
* loss_type = weighted_mse
* defect_weight = 10.0
* lambda_tv = 0
* epoch = 100
* physics_loss：未启用
* L-BFGS：未启用

输出模型：

checkpoints/best_model_v4_balanced_complex_smallpoly_loss.pt

关键 test 结果：

* MSE = 4.10216735e+04
* MAE = 7.83255570e+01
* IoU = 3.22104979e-01
* Dice = 4.67866207e-01
* area_error = 1.34222578e+00
* center_error = 1.14444251e+00

small polygon：

* IoU = 1.36334593e-01
* Dice = 2.26148223e-01
* pred_area = 0 的样本数 = 0 / 25

当前判断：

* weighted MSE 能缓解 small polygon 全部漏检；
* 但 MSE、MAE、area_error 明显变差；
* 当前不切换 CURRENT_BASELINE.md；
* 当前推荐 v3_complex 模型仍为 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt。

## 下一步建议

第 7.12B 步：进行 defect_weight 小范围扫描。

建议候选值：

* 5
* 10
* 20
* 50

暂时不要加入 soft Dice、oversampling、physics_loss、L-BFGS 或模型结构改动。主要基于 val 集选择，再用 test 集做阶段性确认。

---

## 当前最新状态：第 7.12B 步已完成

已完成 v4_balanced_complex 的 defect_weight 扫描。

扫描配置：

* loss_type = weighted_mse
* lambda_tv = 0
* epoch = 100
* physics_loss：未启用
* L-BFGS：未启用
* Dice Loss：未启用
* oversampling：未启用
* 模型结构：未修改

候选值：

* 2
* 3
* 5
* 7
* 10

输出汇总：

* results/metrics/v4_smallpoly_defect_weight_sweep.csv
* results/summaries/v4_smallpoly_defect_weight_sweep_summary.txt

当前推荐 v4 small polygon weighted MSE 候选：

checkpoints/best_model_v4_smallpoly_w5.pt

对应 defect_weight = 5。

关键 test 结果：

* MSE = 3.12321945e+04
* MAE = 6.23678583e+01
* IoU = 3.39080635e-01
* Dice = 4.77603301e-01
* area_error = 8.38023859e-01
* center_error = 1.17307553e+00
* small polygon IoU = 6.54854895e-02
* small polygon Dice = 1.04442883e-01
* small polygon pred_area=0：12 / 25

当前判断：

* defect_weight=5 比 defect_weight=10 的 area_error 明显更低；
* defect_weight=5 仍能让 small polygon 出现有效重叠检出；
* 但该模型尚不足以替代当前 v3_complex 推荐 baseline；
* CURRENT_BASELINE.md 不切换。

## 下一步建议

建议先让 Claude Code review 第 7.12A / 7.12B 的实现和结果记录。

暂不直接进入 Dice Loss。若 review 通过，再考虑以 defect_weight=5 为基础讨论 soft Dice / focal 类 loss。

---

## 当前最新状态：第 7.13 步已完成

已完成 weighted MSE + soft Dice Loss 实验。

训练配置：

* 数据集：v4_balanced_complex
* loss_type = weighted_mse_dice
* defect_weight = 5
* lambda_dice = 0.05
* lambda_tv = 0
* epoch = 100
* physics_loss：未启用
* L-BFGS：未启用
* oversampling：未启用
* 模型结构：未修改

输出模型：

checkpoints/best_model_v4_smallpoly_w5_dice.pt

关键 test 结果：

* MSE = 3.56734905e+04
* MAE = 6.02042826e+01
* IoU = 3.25826098e-01
* Dice = 4.64347405e-01
* area_error = 6.12110696e-01
* center_error = 1.24440727e+00
* small polygon IoU = 1.26014768e-01
* small polygon Dice = 2.01116176e-01
* small polygon pred_area=0：0 / 25
* multi_defect center_error = 1.15517406e+00

当前判断：

* soft Dice 明显减少 small polygon 漏检；
* small polygon IoU / Dice 提升；
* area_error 改善；
* 但 overall IoU / Dice 下降，multi_defect center_error 变差；
* 当前不切换 CURRENT_BASELINE.md；
* 当前 v4 small polygon 候选仍需继续验证，不直接替代 v3_complex 推荐模型。

## 下一步建议

第 7.14 步：lambda_dice 小范围扫描。

建议候选值：

* 0.01
* 0.03
* 0.05
* 0.1

建议继续固定：

* defect_weight = 5
* lambda_tv = 0
* 不启用 physics_loss
* 不启用 L-BFGS
* 不做 oversampling
* 不修改模型结构

---

## 当前最新状态：第 7.13B 步已完成

v4_balanced_complex 的 `lambda_dice` 扫描已完成。

固定配置：
* dataset = v4_balanced_complex
* loss_type = weighted_mse_dice
* defect_weight = 5
* lambda_tv = 0
* epochs = 100
* physics_loss / L-BFGS / oversampling 均未启用

本轮推荐 v4 small polygon 专项候选：

`checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt`

对应 `lambda_dice = 0.03`。

关键结论：
* small polygon pred_area=0 = 0 / 25；
* overall IoU / Dice 相比 weighted MSE w5 恢复并提升；
* multi_defect center_error 相比 weighted MSE w5 改善；
* area_error 仍偏大，因此不切换全项目 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

## 下一步建议

先不要直接进入 focal loss 或 oversampling。建议优先分析 `lambda_dice=0.03` 下预测面积偏大的原因，或设计更温和的面积约束 / 后处理验证方案。
---

## 当前最新状态：第 7.14 步已完成

已完成 `lambda_dice=0.03` 模型的 area_error 诊断。

诊断模型：

`checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt`

关键结论：
* pred_area 系统性大于 true_area：190 / 200 个样本为面积高估；
* area_error 主要集中在 polygon；
* small polygon 的 IoU / Dice 最低，但 mean area_error 最高的是 medium polygon；
* worst10 中 9 个是 polygon，1 个是 ellipse；
* multi_defect 不是本轮 area_error 主因；
* 当前不切换全项目 baseline。

输出摘要：

`results/summaries/v4_smallpoly_area_error_diagnosis_summary.txt`

## 下一步建议

不要直接进入 focal loss 或 oversampling。建议优先做轻量验证：

1. 固定 `loss_type=weighted_mse_dice`；
2. 固定 `lambda_dice=0.03`；
3. 尝试降低 `defect_weight` 到 3 或 4；
4. 观察 small polygon 漏检是否仍为 0，同时 area_error 是否下降。

如果降低 defect_weight 仍不能解决过分割，再考虑加入可选 `area-aware loss`，例如新增 `lambda_area`，基于 soft mask 面积做相对面积约束。
---

## 当前最新状态：第 7.15 步已完成

已完成 v4_balanced_complex 的 area-aware loss 面积约束实验。

本轮推荐 v4 area-aware 专项候选：

`checkpoints/best_model_v4_smallpoly_w5_dice_area_0p05.pt`

对应配置：
* loss_type = weighted_mse_dice_area
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.05
* lambda_tv = 0
* physics_loss / L-BFGS / oversampling 均未启用

关键结论：
* overall area_error 从无 area loss 的 1.002027 降到 0.794285；
* polygon area_error 从 1.478094 降到 1.197988；
* medium polygon area_error 从 1.891204 降到 1.429367；
* small polygon pred_area=0 仍为 0 / 25；
* overall IoU / Dice 轻微下降；
* multi_defect center_error 略有变差；
* 当前不切换全项目 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

## 下一步建议

建议先让 Claude Code review 第 7.15 步实现与结果。后续若继续优化，优先考虑：

1. 围绕 `lambda_area=0.05` 做小范围复核，例如 `0.04 / 0.05 / 0.07`；
2. 或验证 `defect_weight=3 / 4` 是否能进一步降低过分割；
3. 暂不建议直接进入 focal loss 或 oversampling。
---

## 当前最新状态：第 7.16 步已完成

已完成面积约束细化实验。

主要结果：
* symmetric `lambda_area=0.07` 是 symmetric 细扫中面积指标最好的候选；
* `over_only lambda_area=0.05` 更能降低 `pred_area > true_area` 数量，从约 185-189 / 200 降到 166 / 200；
* 但 `over_only` 使 small polygon pred_area=0 回升到 14 / 25，不适合作为当前方案；
* 当前不切换全项目 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

## 下一步建议

建议先让 Claude Code review 第 7.16 步实现和结果。若继续实验，优先考虑：

1. `defect_weight=3 / 4` 与 symmetric `lambda_area=0.04 / 0.07` 的组合；
2. 或暂停 loss 调参，转向误差样本可视化和模型结构方案讨论；
3. 暂不建议进入 focal loss 或 oversampling。
---

## 当前最新状态：第 7.17 步已完成

已完成 v4_balanced_complex 的 symmetric area loss 组合验证。

固定配置：

* dataset = v4_balanced_complex
* loss_type = weighted_mse_dice_area
* lambda_dice = 0.03
* lambda_tv = 0
* area_loss_type = symmetric
* epochs = 100

验证组合：

* defect_weight = 5, lambda_area = 0.04
* defect_weight = 5, lambda_area = 0.07
* defect_weight = 7, lambda_area = 0.04
* defect_weight = 7, lambda_area = 0.07

本轮综合表现最好的 v4 small polygon / area loss 候选为：

`checkpoints/best_model_v4_w5_dice003_area004.pt`

对应配置：

* defect_weight = 5
* lambda_area = 0.04

关键结论：

* small polygon pred_area=0 仍为 0 / 25；
* overall IoU / Dice 在本轮四组中最好；
* multi_defect center_error 在本轮四组中最低；
* polygon area_error 没有继续下降，且不如第 7.16 步的 symmetric lambda_area=0.07；
* 当前仍不切换全项目 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

## 下一步建议

暂不继续扩大 loss 调参，也不直接进入 focal loss 或 oversampling。建议先围绕模型结构或后处理方案制定下一阶段计划；如果继续比较 loss，需要加入 seed / repeat 验证来降低训练随机性影响。
---

## 当前最新状态：第 7.18 步已完成

已完成 v4 small polygon / area-aware 候选模型的后处理与阈值分析。

分析模型：

`checkpoints/best_model_v4_w5_dice003_area004.pt`

分析数据集：

`data/training_data_v4_balanced_complex_test.npz`

主要结论：

* 标准 threshold=500 时，area_error = 0.911511，pred_area > true_area = 191 / 200；
* threshold=300 时，area_error 降至 0.292975，pred_area > true_area 降至 114 / 200；
* threshold=300 下 small polygon `pred_area=0` 仍为 0 / 25；
* IoU / Dice 最优更接近 threshold=450；
* 连通域过滤 remove < 5 / 10 / 20 pixels 基本没有额外收益；
* 后处理可作为可选评估方案，但不替代标准评价指标，也不切换 baseline。

输出文件：

* `results/metrics/v4_postprocess_threshold_sweep.csv`
* `results/metrics/v4_postprocess_component_filter.csv`
* `results/summaries/v4_postprocess_analysis_summary.txt`
* `results/previews/v4_postprocess_examples/`

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

## 下一步建议

进入第 7.19 步：模型结构优化方案设计。建议先制定方案，不要直接大改代码。重点考虑：

1. 多尺度坐标特征；
2. 更强的 BzEncoder；
3. polygon 边界表达能力；
4. 小目标缺陷的 mask 解码能力；
5. 为 `train_pinn.py` 增加 `--seed` 参数，提高后续实验可复现性。
---

## 当前最新状态：第 7.18.5 步已完成

已完成 `train_pinn.py` 的训练随机种子支持。

新增能力：

* `--seed` 参数；
* 默认 `seed = 42`；
* 训练启动时打印当前 seed；
* Adam 训练的 shuffle DataLoader 使用固定 `torch.Generator()`；
* `set_seed(seed)` 同步设置 Python random、NumPy、PyTorch 和 CUDA 随机种子。

说明：

第 7.15–7.17 的结果显示，相同配置存在训练随机性波动。因此从第 7.18.5 开始，后续模型结构优化实验默认固定 `seed=42`。

第 7.18 后处理阈值分析还说明，模型预测 μ 值存在校准偏软问题：缺陷区域常预测为 μ≈200–400，而不是接近真实 μ≈1。因此 threshold=300 能显著降低 area_error。这说明问题不只是评估阈值，而是模型输出校准和边界表达能力不足。

## 下一步建议

进入第 7.19 步：模型结构优化方案设计。先制定方案，不直接大改代码。后续结构对比必须固定 `--seed 42`，关键结论建议做 repeat 验证。
