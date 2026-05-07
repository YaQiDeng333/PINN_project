NEXT_STEP

当前状态（最新）

第 7.7 步：v3 complex 延长训练与专用 lambda_tv 扫描已完成。

当前推荐 v3 complex 模型

checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt

推荐配置

mode = adam_tv

epochs = 50

lambda_tv = 2e-6

lambda_phy = 0

L-BFGS = disabled

推荐依据

该模型由 v3_complex 专用 lambda_tv 扫描选出，选择依据是 val_iou、val_dice、val_mae 综合排序。

test 集确认结果显示，相比第 7.5 步 20 epoch v3 baseline：

MSE、IoU、Dice、area_error、center_error 改善；

MAE 变差。

第 7.7 步关键输出

results/summaries/v3_complex_long_training_summary.txt

results/summaries/v3_complex_lambda_tv_sweep_summary.txt

results/metrics/evaluation_metrics_v3_complex_tv_long.csv

results/metrics/v3_complex_long_metrics_by_type.csv

results/metrics/v3_complex_lambda_tv_sweep.csv

results/metrics/evaluation_metrics_v3_complex_tv_sweep_2e-6_test.csv

results/metrics/v3_complex_sweep_2e-6_test_metrics_by_type.csv

当前判断

100 epoch 长训练没有成为默认推荐：它只改善 MSE 和 center_error，但 MAE、IoU、Dice、area_error 变差。

lambda_tv=2e-6 的 sweep 模型改善了整体 IoU / Dice，并明显改善 polygon 的 IoU / Dice。

multi_defect 仍然困难：MSE / MAE 和 mask 类指标没有稳定改善，只有 center_error 略好。

当前下一步

暂不进入 physics_loss、L-BFGS 或模型结构大改。

建议先围绕当前推荐 v3_complex 模型做更细的误差诊断，重点看：

1. polygon 仍然漏检的样本；
2. multi_defect 的多目标分离失败样本；
3. 是否需要针对复杂缺陷调整训练采样或 loss 权重。

如果后续确认训练策略已经到瓶颈，再进入模型结构优化。
