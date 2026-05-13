# CURRENT_BASELINE

## 当前最佳模型

checkpoints/best_model_tv_5e-6.pt

## 当前推荐配置

推荐 lambda_tv = 5e-6

当前是否使用 L-BFGS：否。

L-BFGS 仅作为可选实验保留，不作为默认推荐方案。

## 当前 test 指标

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

## 第 7.7 步后当前 v3 complex 推荐 baseline

以本节记录为准，当前 v3 complex 推荐模型已更新为：

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

## 第 7.7 步更新说明

相比第 7.5 步 20 epoch v3 baseline，当前推荐模型改善了 MSE、IoU、Dice、area_error、center_error，但 MAE 变差。

simple baseline 仍为 checkpoints/best_model_tv_5e-6.pt。

v3 complex 推荐 baseline 更新为 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt。

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
