NEXT_STEP

当前状态

正式规模 v3 complex 复杂缺陷数据集已生成并通过检查。

当前默认最佳 baseline

默认 baseline 模型：

checkpoints/best_model_tv_5e-6.pt

默认 lambda_tv：

5e-6

默认 lambda_phy：

0

默认是否启用 L-BFGS：

否。L-BFGS 仅保留为 optional refine 实验，不作为默认推荐方案。

v3 complex 数据集

已生成：

data/training_data_v3_complex_train.npz

data/training_data_v3_complex_val.npz

data/training_data_v3_complex_test.npz

样本数量：

train = 1000

val = 200

test = 200

检查摘要：

results/summaries/v3_complex_dataset_summary.txt

defect_types 分布

train：

multi_defect = 331

polygon = 348

rotated_rect = 321

val：

multi_defect = 71

polygon = 72

rotated_rect = 57

test：

multi_defect = 62

polygon = 75

rotated_rect = 63

验证结论

三个 npz 均存在。

样本数量正确。

signals / mu_maps / metadata / metadata_keys 正常。

每个样本缺陷 mask 非空。

signals 和 mu_maps 无 NaN / Inf。

旧 simple 数据集未覆盖。

当前下一步

建议进入 v3 complex 模型训练阶段，但需等待用户确认。

建议要求：

1. 不覆盖当前 simple baseline 模型；
2. 使用新的 checkpoint 文件名，例如 checkpoints/best_model_v3_complex_tv.pt；
3. 使用新的 results 输出前缀或文件名；
4. 主要基于 v3 complex val 集选择参数；
5. v3 complex test 集只用于阶段性最终评估；
6. simple baseline 和 complex baseline 分开记录。

执行约束

不要覆盖旧数据集：

data/training_data_train.npz

data/training_data_val.npz

data/training_data_test.npz

不要覆盖旧 checkpoints。

不要修改 evaluate_pinn.py 的评价指标逻辑。

如果修改 train_pinn.py，应保持原 simple 数据训练流程可用。
