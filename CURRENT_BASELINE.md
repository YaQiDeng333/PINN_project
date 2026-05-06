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
