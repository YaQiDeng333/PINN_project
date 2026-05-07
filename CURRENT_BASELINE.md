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
