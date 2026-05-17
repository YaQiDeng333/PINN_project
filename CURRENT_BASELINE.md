# CURRENT_BASELINE

## 当前权威 boundary baseline（第 18.4 后）

本节是当前 `CURRENT_BASELINE` 的权威记录。若下方历史记录与本节冲突，以本节为准。

当前 boundary-oriented `CURRENT_BASELINE` 已更新为 v3_complex mask-only grid decoder + forward consistency：

* model family：mask-only grid decoder + forward consistency
* dataset：`v3_complex`
* forward surrogate：`checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`
* `lambda_forward = 0.10`
* selected probability threshold：`0.80`
* threshold selection source：仅 validation set
* checkpoint selection：validation IoU + validation Dice - validation area_error
* main evaluation：test set，`mask_prob >= 0.80`
* loss：BCEWithLogits + soft Dice + `0.10 * MSE(frozen_surrogate(mask_prob), Bz_obs)`

本项目目标是缺陷边界形状反演。该 baseline 的主要优势是 shape metrics 与 Bz consistency 同时改善：IoU、Dice、area_error、center_error 和 Bz MSE 均优于上一版 mask-only grid decoder baseline，且 `pred_area=0` 没有恶化。

### 当前 baseline test 指标

以下数值为 3 seed mean +/- sample std，测试集为 `data/training_data_v3_complex_test.npz`。

| metric | value |
|---|---:|
| IoU | 0.3563 +/- 0.0064 |
| Dice | 0.5017 +/- 0.0089 |
| area_error | 0.2734 +/- 0.0226 |
| center_error | 1.1464 +/- 0.0302 |
| Bz MSE | 0.1649 +/- 0.0135 |
| pred_area=0 | 1.33 +/- 1.15 |
| MSE / MAE | N/A for full mu field |

### 分组指标与已知 trade-off

| group | IoU | Dice | area_error | Bz MSE |
|---|---:|---:|---:|---:|
| small | 0.2909 +/- 0.0036 | 0.4210 +/- 0.0037 | 0.4193 +/- 0.0552 | 0.1702 +/- 0.0241 |
| low_signal | 0.2706 +/- 0.0039 | 0.4025 +/- 0.0049 | 0.3788 +/- 0.0287 | 0.1861 +/- 0.0185 |
| polygon | 0.3151 +/- 0.0008 | 0.4490 +/- 0.0005 | 0.3743 +/- 0.0407 | 0.1480 +/- 0.0230 |
| rotated_rect | 0.3968 +/- 0.0092 | 0.5482 +/- 0.0126 | 0.1985 +/- 0.0089 | 0.1339 +/- 0.0184 |

需要明确保留的 trade-off：polygon area_error 相比上一版 grid baseline 轻微恶化（`0.3563 +/- 0.0565` -> `0.3743 +/- 0.0407`），polygon / rotated_rect 精细边界圆斑化问题仍未完全解决。第 18.4 的主要收益是整体 shape 指标和 Bz consistency 同时改善，不是彻底解决 polygon 细边界。

### 保留参考线

上一版 mask-only grid decoder + threshold=0.90 保留为 boundary reference：

* `checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed42.pt`
* `checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed123.pt`
* `checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed2026.pt`
* threshold：`0.90`
* test metrics：IoU `0.33909 +/- 0.00483`，Dice `0.48120 +/- 0.00413`，area_error `0.28853 +/- 0.01164`，center_error `1.24894 +/- 0.01191`，pred_area=0 `1.33 +/- 1.15`，Bz MSE `0.3323 +/- 0.0149`

更早的 mask-only MLP boundary baseline 保留为 earlier boundary reference：

* `checkpoints/mask_boundary_candidate/best_mask_boundary_seed42.pt`
* `checkpoints/mask_boundary_candidate/best_mask_boundary_seed123.pt`
* `checkpoints/mask_boundary_candidate/best_mask_boundary_seed2026.pt`
* threshold: `0.90`
* test metrics: IoU `0.3319 +/- 0.0169`, Dice `0.4729 +/- 0.0189`, area_error `0.3220 +/- 0.0087`, center_error `1.2271 +/- 0.0083`, pred_area=0 `3.67 +/- 0.58`

The composite-selection baseline is retained as a mu-threshold shape-oriented reference:

* `checkpoints/best_model_v3_complex_composite_seed42.pt`
* `checkpoints/best_model_v3_complex_composite_seed123.pt`
* `checkpoints/best_model_v3_complex_composite_seed2026.pt`
* threshold: raw `mu < 500`
* its MSE / MAE and mask metrics are 3 seed means, not single-checkpoint metrics.

The old `v3_complex_tv_sweep_2e-6` model is retained as an MSE-oriented reference:

* `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

### 已知剩余问题

forward consistency 改善了当前 overall boundary metrics 和 Bz consistency，但没有完全解决精细边界形状。polygon 和 rotated_rect 样本仍可能出现圆斑化 / 平滑斑块化，因此不能把该 baseline 理解为边界问题已经完成解决。

下方历史 baseline 记录仅保留用于溯源。若下方内容与第 18.4 当前权威 baseline 冲突，以第 18.4 记录为准。

## 当前 boundary-oriented CURRENT_BASELINE（第 15.1 / 15.2 后，以此为准）

本项目当前目标是缺陷边界形状反演，因此 CURRENT_BASELINE 以 IoU、Dice、area_error、`pred_area=0`、small / low-signal 表现为主要依据，而不是只选择 MSE / MAE 最低的 μ field 模型。

当前 boundary-oriented baseline 是 v3_complex mask-only boundary model。该模型不预测完整 μ field，而是直接从 Bz signal + 空间坐标预测 defect probability / mask。第 15.2 使用 validation set only 选择全局 probability threshold = 0.90，test set 只用于最终验证。

### 当前 baseline checkpoint family

本轮以 3 seed mask-only candidate set 作为当前 boundary-oriented baseline 记录：

* `checkpoints/mask_boundary_candidate/best_mask_boundary_seed42.pt`
* `checkpoints/mask_boundary_candidate/best_mask_boundary_seed123.pt`
* `checkpoints/mask_boundary_candidate/best_mask_boundary_seed2026.pt`

对应数据集：

* train：`data/training_data_v3_complex_train.npz`
* val：`data/training_data_v3_complex_val.npz`
* test：`data/training_data_v3_complex_test.npz`

训练 / 选择配置：

* `dataset = v3_complex`
* model family = mask-only boundary model
* target = `target_mu_norm < 0.5`，等价 raw `target_mu < 500`
* loss = BCEWithLogits + soft Dice
* best checkpoint selection = validation IoU + validation Dice - validation area_error
* selected probability threshold = 0.90
* threshold selection source = validation set only
* main evaluation = test set, `mask_prob >= 0.90`
* physics_loss：否
* L-BFGS：否

### 当前 baseline test 指标（mask_prob >= 0.90，3 seed mean +/- sample std）

| 指标 | 数值 |
|---|---:|
| IoU | 3.319e-01 +/- 1.69e-02 |
| Dice | 4.729e-01 +/- 1.89e-02 |
| area_error | 3.220e-01 +/- 8.71e-03 |
| center_error | 1.227e+00 +/- 8.31e-03 |
| pred_area=0 | 3.67 +/- 0.58 |
| MSE | N/A |
| MAE | N/A |

MSE / MAE 不适合作为该 baseline 的主指标，因为 mask-only model 不预测完整 μ field。

### 当前 baseline small / low-signal 关键指标（mask_prob >= 0.90，3 seed mean +/- sample std）

| 分组 | IoU | Dice | area_error |
|---|---:|---:|---:|
| small | 2.743e-01 +/- 1.71e-02 | 3.981e-01 +/- 2.24e-02 | 3.706e-01 +/- 7.15e-02 |
| low_signal | 2.454e-01 +/- 2.11e-02 | 3.673e-01 +/- 2.63e-02 | 4.222e-01 +/- 3.50e-02 |

选择依据：

第 15.1 fixed threshold=0.5 显示 mask-only boundary model 明显提升 IoU / Dice，并显著减少 `pred_area=0`，但存在严重面积高估，因此不能按 0.5 threshold 直接接受。第 15.2 仅用 validation set 选择 probability threshold=0.90 后，test set 上 IoU / Dice 仍高于 composite-selection baseline，area_error 被压低到略优于 composite-selection baseline，`pred_area=0` 明显减少；small 与 low-signal 样本的 IoU / Dice 也保持改善。因此 mask-only + threshold=0.90 更符合当前“缺陷边界形状反演”的主目标。

### μ-threshold shape-oriented reference baseline（保留对照）

原 composite-selection baseline 保留为 μ-threshold shape-oriented reference，而不是当前 boundary-oriented CURRENT_BASELINE：

* `checkpoints/best_model_v3_complex_composite_seed42.pt`
* `checkpoints/best_model_v3_complex_composite_seed123.pt`
* `checkpoints/best_model_v3_complex_composite_seed2026.pt`

其 threshold=500 test 指标为：

| 指标 | 数值 |
|---|---:|
| MSE | 2.1444e+04 +/- 2.72e+02 |
| MAE | 4.9181e+01 +/- 1.39e+00 |
| IoU | 3.2166e-01 +/- 7.28e-03 |
| Dice | 4.5455e-01 +/- 8.68e-03 |
| area_error | 3.3744e-01 +/- 1.95e-02 |
| center_error | 1.2257e+00 +/- 1.23e-02 |
| pred_area=0 | 10.33 +/- 5.13 |

上述 MSE / MAE 以及其他指标均为 3 seed mean，而不是单个 checkpoint 指标。

该 baseline 仍适合作为 μ-threshold shape-oriented reference，尤其用于对照完整 μ field 模型的 mask-threshold 表现。

### MSE-oriented reference baseline（保留对照）

旧 v3_complex 推荐模型保留为 MSE-oriented reference baseline，而不是当前 boundary-oriented CURRENT_BASELINE：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

其 test 指标为：

| 指标 | 数值 |
|---|---:|
| MSE | 2.07377174e+04 |
| MAE | 4.44655262e+01 |
| IoU | 2.95272047e-01 |
| Dice | 4.21885407e-01 |
| area_error | 3.94517442e-01 |
| center_error | 1.32594189e+00 |

该模型仍适合作为 MSE / MAE 数值误差参考，但不再代表当前 boundary-oriented 主线最佳模型。

---

## simple 缺陷历史最佳模型（保留参考）

checkpoints/best_model_tv_5e-6.pt

## simple 缺陷推荐配置（历史参考）

推荐 lambda_tv = 5e-6

当前是否使用 L-BFGS：否。

L-BFGS 仅作为可选实验保留，不作为默认推荐方案。

## simple 缺陷 test 指标（历史参考）

MSE = 2.16568206e+04

MAE = 4.39399008e+01

IoU = 4.32040206e-01

Dice = 5.82132493e-01

area_error = 2.42350201e-01

center_error = 1.03291037e+00

## 第六步 baseline

第六步物理一致性 Loss 将以 checkpoints/best_model_tv_5e-6.pt 和 lambda_tv=5e-6 作为 baseline。

默认不启用 L-BFGS。

## 第六步初版实验后结论

checkpoints/best_model_tv_phy.pt 已完成训练和评估，但除 MSE 略有改善外，MAE、IoU、Dice、area_error、center_error 均差于当前最佳 TV baseline。

因此当前最佳模型仍为 checkpoints/best_model_tv_5e-6.pt。

## 第 7.5 步原始 v3 complex 复杂缺陷 baseline

复杂缺陷 baseline 模型：

checkpoints/best_model_v3_complex_tv.pt

训练数据：

data/training_data_v3_complex_train.npz

验证数据：

data/training_data_v3_complex_val.npz

测试数据：

data/training_data_v3_complex_test.npz

推荐 lambda_tv = 5e-6

当前是否使用 physics_loss：否。

当前是否使用 L-BFGS：否。

## v3 complex test 指标

MSE = 2.07475147e+04

MAE = 4.36197426e+01

IoU = 2.76481934e-01

Dice = 3.97991681e-01

area_error = 4.26162950e-01

center_error = 1.34338298e+00

## baseline 区分说明

simple baseline 仍为 checkpoints/best_model_tv_5e-6.pt。

第 7.5 步原始 v3 complex baseline 为 checkpoints/best_model_v3_complex_tv.pt。

第 7.7 步后的当前 v3 complex 推荐 baseline 见下节。

两者对应的数据集难度不同，不应直接混用为同一个 baseline。

## 第 7.7 步后 v3 complex MSE-oriented reference baseline

第 13.4 / 13.5 前，本节模型曾作为 v3_complex 推荐 baseline。第 13.4 / 13.5 后，它保留为 MSE-oriented reference baseline：

checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt

训练数据：
data/training_data_v3_complex_train.npz

验证数据：
data/training_data_v3_complex_val.npz

测试数据：
data/training_data_v3_complex_test.npz

推荐 lambda_tv = 2e-6

当前是否使用 physics_loss：否。

当前是否使用 L-BFGS：否。

选择依据：
v3_complex 专用 lambda_tv 扫描中，lambda_tv=2e-6 按 val_iou、val_dice、val_mae 综合排序最佳。

## 第 7.7 步后 v3 complex test 指标

MSE = 2.07377174e+04

MAE = 4.44655262e+01

IoU = 2.95272047e-01

Dice = 4.21885407e-01

area_error = 3.94517442e-01

center_error = 1.32594189e+00

## 第 7.7 步更新说明（历史记录）

相比第 7.5 步 20 epoch v3 baseline，当前推荐模型改善了 MSE、IoU、Dice、area_error、center_error，但 MAE 变差。

simple baseline 仍为 checkpoints/best_model_tv_5e-6.pt。

v3 complex 推荐 baseline 当时更新为 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt。第 13.4 / 13.5 后，该模型保留为 MSE-oriented reference baseline。

## 第 7.9 步 v4 balanced complex 正式数据集

当前 v4 balanced complex 是正式规模数据集，但尚未训练对应模型，因此不是训练 baseline。

数据文件：

data/training_data_v4_balanced_complex_train.npz

data/training_data_v4_balanced_complex_val.npz

data/training_data_v4_balanced_complex_test.npz

样本数量：

train = 1000

val = 200

test = 200

生成 seed：

7904

当前检查摘要：

results/summaries/v4_balanced_complex_dataset_summary.txt

v4 数据生成规则说明：

polygon area_bin 阈值为 small < 120 pixels，120 <= medium < 500 pixels，large >= 500 pixels。

polygon 生成保留 mask_pixels >= 30 和 signal_snr >= 5 检查。

multi_defect 中 2 缺陷 / 3 缺陷约按 40% / 60% 分配。

正式 train 检查结果：

defect_types：

circle = 75

ellipse = 75

multi_defect = 300

polygon = 300

rect = 75

rotated_rect = 100

triangle = 75

polygon area_bin：

small = 124

medium = 103

large = 73

multi_defect num_defects：

2 defects = 120

3 defects = 180

检查结论：

metadata_keys 与 metadata 字段一致。

signals 和 mu_maps 无 NaN / Inf。

每个样本缺陷 mask 非空。

polygon 没有低于 mask_pixels 或 signal_snr 阈值的样本。

说明：

由于 data_generator_v2.py 的 metadata dtype 是全局定义，如果后续重新生成 simple 或 v3_complex 数据集，新生成的 npz 也可能包含 mask_pixels、signal_snr、area_bin 等 v4 新增 metadata 字段。旧数据集文件本身没有被覆盖。

## 第 7.12A 步记录：v4 small polygon weighted MSE 实验

本节仅记录实验结果，不切换当前推荐 baseline。

实验模型：

checkpoints/best_model_v4_balanced_complex_smallpoly_loss.pt

训练数据：

data/training_data_v4_balanced_complex_train.npz

验证数据：

data/training_data_v4_balanced_complex_val.npz

测试数据：

data/training_data_v4_balanced_complex_test.npz

训练配置：

loss_type = weighted_mse

defect_weight = 10.0

lambda_tv = 0

physics_loss：否。

L-BFGS：否。

soft Dice：否。

oversampling：否。

v4 test 指标：

MSE = 4.10216735e+04

MAE = 7.83255570e+01

IoU = 3.22104979e-01

Dice = 4.67866207e-01

area_error = 1.34222578e+00

center_error = 1.14444251e+00

small polygon：

IoU = 1.36334593e-01

Dice = 2.26148223e-01

pred_area = 0 的样本数 = 0 / 25

结论：

weighted MSE 能缓解 small polygon 全部漏检，但 MSE、MAE、area_error 明显变差。当前不切换 baseline。

当前 v3_complex 推荐模型仍为：

checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt

## 第 7.13 步记录：v4 weighted MSE + soft Dice Loss

本节仅记录实验结果，不切换当前推荐 baseline。

实验模型：

checkpoints/best_model_v4_smallpoly_w5_dice.pt

训练配置：

loss_type = weighted_mse_dice

defect_weight = 5

lambda_dice = 0.05

lambda_tv = 0

physics_loss：否。

L-BFGS：否。

oversampling：否。

v4 test 指标：

MSE = 3.56734905e+04

MAE = 6.02042826e+01

IoU = 3.25826098e-01

Dice = 4.64347405e-01

area_error = 6.12110696e-01

center_error = 1.24440727e+00

small polygon：

IoU = 1.26014768e-01

Dice = 2.01116176e-01

pred_area = 0 的样本数 = 0 / 25

multi_defect center_error = 1.15517406e+00

结论：

soft Dice 明显改善 small polygon 漏检，并降低 area_error；但 overall IoU / Dice 下降，multi_defect center_error 变差。当前不切换全项目 baseline。

当前 v3_complex 推荐模型仍为：

checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt

## 第 7.12B 步记录：v4 small polygon defect_weight 扫描

本节仅记录实验结果，不切换当前推荐 baseline。

扫描候选：

2、3、5、7、10

训练配置：

loss_type = weighted_mse

lambda_tv = 0

epoch = 100

physics_loss：否。

L-BFGS：否。

Dice Loss：否。

oversampling：否。

当前 v4 small polygon weighted MSE 推荐候选：

checkpoints/best_model_v4_smallpoly_w5.pt

defect_weight = 5

v4 test 指标：

MSE = 3.12321945e+04

MAE = 6.23678583e+01

IoU = 3.39080635e-01

Dice = 4.77603301e-01

area_error = 8.38023859e-01

center_error = 1.17307553e+00

small polygon：

IoU = 6.54854895e-02

Dice = 1.04442883e-01

pred_area = 0 的样本数 = 12 / 25

结论：

defect_weight=5 比 defect_weight=10 的面积误差明显更低，同时 small polygon 不再全部漏检。当前仍不切换全项目 baseline。

当前 v3_complex 推荐模型仍为：

checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt

---

## 第 7.13B 步记录：v4 small polygon lambda_dice 扫描

本节仅记录实验结果，不切换当前推荐 baseline。

推荐的 v4 small polygon 专项候选：

`checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt`

配置：
* dataset = v4_balanced_complex
* loss_type = weighted_mse_dice
* defect_weight = 5
* lambda_dice = 0.03
* lambda_tv = 0
* physics_loss：否
* L-BFGS：否
* oversampling：否

v4 test 指标：
* MSE = 3.36594865e+04
* MAE = 6.24013944e+01
* IoU = 3.52031766e-01
* Dice = 4.97474011e-01
* area_error = 1.00202667e+00
* center_error = 1.11096996e+00
* small polygon pred_area=0 = 0 / 25
* multi_defect center_error = 8.77950897e-01

结论：该候选改善了 small polygon 漏检、overall IoU / Dice 和 multi_defect center_error，但 area_error 仍偏大，因此不替换当前全项目推荐 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

---

## 第 7.20A 步记录：calibrated_mu 输出 μ 参数化校准

本节仅记录实验结果，不切换当前全项目推荐 baseline。

实验数据集：

`data/training_data_v4_balanced_complex_train.npz`

`data/training_data_v4_balanced_complex_val.npz`

`data/training_data_v4_balanced_complex_test.npz`

固定配置：

* seed = 42
* loss_type = weighted_mse_dice_area
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.04
* area_loss_type = symmetric
* lambda_tv = 0

对比模型：

* baseline：`checkpoints/best_model_v4_baseline_seed42_w5_dice003_area004.pt`
* calibrated_mu：`checkpoints/best_model_v4_calibrated_mu_seed42_w5_dice003_area004.pt`

calibrated_mu test 指标：

* MSE = 3.06620221e+04
* MAE = 5.87495580e+01
* IoU = 3.54232016e-01
* Dice = 4.96098795e-01
* area_error = 6.40109928e-01
* center_error = 1.13673565e+00

实验结论：

`calibrated_mu` 相比 baseline seed=42 改善了 IoU、Dice、center_error、polygon area_error 和 small polygon 指标，缺陷区预测 μ_r 均值也从约 399 降到约 361。但 area_error 几乎不变，`pred_area > true_area` 数量增加到 182 / 200，因此不切换当前全项目 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

## 第 7.14 步记录：area_error 诊断

本节仅记录诊断结果，不切换当前推荐 baseline。

诊断模型：

`checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt`

关键结论：
* mean area_ratio = 1.989118
* median area_ratio = 1.806303
* pred_area > true_area = 190 / 200
* area_error 主要集中在 polygon；
* small polygon 的 IoU / Dice 最低，但 mean area_error 最高的是 medium polygon；
* worst10 中 9 个是 polygon；
* multi_defect 不是本轮 area_error 主因。

结论：`lambda_dice=0.03` 仍是 v4 small polygon 专项候选，但 area_error 偏大，不能切换为全项目推荐 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
---

## 第 7.15 步记录：area-aware loss 面积约束实验

本节仅记录实验结果，不切换当前推荐 baseline。

推荐的 v4 area-aware 专项候选：

`checkpoints/best_model_v4_smallpoly_w5_dice_area_0p05.pt`

配置：
* dataset = v4_balanced_complex
* loss_type = weighted_mse_dice_area
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.05
* lambda_tv = 0
* physics_loss：否
* L-BFGS：否
* oversampling：否

v4 test 指标：
* MSE = 3.20781063e+04
* MAE = 6.21932516e+01
* IoU = 3.50964003e-01
* Dice = 4.94989266e-01
* area_error = 7.94284719e-01
* center_error = 1.24532434e+00
* polygon area_error = 1.197988
* small polygon pred_area=0 = 0 / 25
* medium polygon area_error = 1.429367
* multi_defect center_error = 9.67055063e-01

结论：该候选改善了面积高估和 polygon area_error，但 overall IoU / Dice 略降，multi_defect center_error 略变差，因此不替换当前全项目推荐 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
---

## 第 7.16 步记录：面积约束细化实验

本节仅记录实验结果，不切换当前推荐 baseline。

symmetric 细扫中面积指标最好的候选：

`checkpoints/best_model_v4_smallpoly_w5_dice_area_refine_0p07.pt`

配置：
* loss_type = weighted_mse_dice_area
* area_loss_type = symmetric
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.07
* lambda_tv = 0

v4 test 指标：
* IoU = 3.45279115e-01
* Dice = 4.86960577e-01
* area_error = 6.91137229e-01
* polygon area_error = 9.26742e-01
* small polygon pred_area=0 = 0 / 25
* medium polygon area_error = 1.203071
* multi_defect center_error = 9.92382e-01

`over_only lambda_area=0.05` 虽然将 `pred_area > true_area` 降到 166 / 200，但 small polygon pred_area=0 回升到 14 / 25，因此不作为当前推荐方案。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
---

## 第 7.17 步记录：v4 symmetric area loss 组合验证

本节仅记录实验结果，不切换当前全项目推荐 baseline。

验证配置：

* dataset = v4_balanced_complex
* loss_type = weighted_mse_dice_area
* lambda_dice = 0.03
* lambda_tv = 0
* area_loss_type = symmetric
* epochs = 100

本轮综合表现最好的 v4 small polygon / area loss 候选：

`checkpoints/best_model_v4_w5_dice003_area004.pt`

对应参数：

* defect_weight = 5
* lambda_area = 0.04

v4 test 关键指标：

* IoU = 3.51332857e-01
* Dice = 4.96197269e-01
* area_error = 9.11510936e-01
* polygon area_error = 1.42830383e+00
* small polygon pred_area=0 = 0 / 25
* small polygon IoU = 1.23676006e-01
* small polygon Dice = 2.02804194e-01
* multi_defect center_error = 8.64540981e-01

结论：

该模型在本轮四组中 overall IoU / Dice、small polygon 检出和 multi_defect center_error 综合最好，但 polygon area_error 没有继续下降，也没有明显优于当前全项目推荐 baseline。

当前全项目推荐 baseline 不变：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
---

## 第 7.18 步记录：后处理与阈值分析

本节仅记录诊断结果，不切换当前全项目推荐 baseline。

分析对象：

`checkpoints/best_model_v4_w5_dice003_area004.pt`

分析数据集：

`data/training_data_v4_balanced_complex_test.npz`

标准 threshold=500：

* IoU = 3.51332857e-01
* Dice = 4.96197269e-01
* area_error = 9.11510936e-01
* pred_area > true_area = 191 / 200
* polygon area_error = 1.42830383e+00
* small polygon pred_area=0 = 0 / 25

后处理 threshold=300：

* IoU = 3.37844546e-01
* Dice = 4.75547692e-01
* area_error = 2.92974545e-01
* pred_area > true_area = 114 / 200
* polygon area_error = 3.90190968e-01
* small polygon pred_area=0 = 0 / 25

结论：

降低 mask threshold 可以明显改善面积高估，threshold=300 是本轮 area_error 最优后处理候选；threshold=450 的 IoU / Dice 更高。连通域过滤基本没有额外收益。后处理仅作为可选评估方案，不替代标准评价指标，也不切换当前全项目 baseline。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
---

## 第 7.18.5 步记录：训练随机种子 seed 支持

本节仅记录流程改进，不切换当前全项目推荐 baseline。

`train_pinn.py` 已新增 `--seed` 参数，默认值为 `42`。后续训练启动时会打印当前 seed，并对 Python random、NumPy、PyTorch 和 CUDA 随机种子进行设置。Adam 训练的 shuffle DataLoader 也使用固定 `torch.Generator()`。

该改动用于提高后续第 7.19 模型结构优化实验的可复现性，不改变当前推荐模型。

当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
