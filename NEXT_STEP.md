NEXT_STEP

当前状态

第七步：复杂缺陷扩展第一版已完成。

当前默认最佳 baseline

默认 baseline 模型：

checkpoints/best_model_tv_5e-6.pt

默认 lambda_tv：

5e-6

默认 lambda_phy：

0

默认是否启用 L-BFGS：

否。L-BFGS 仅保留为 optional refine 实验，不作为默认推荐方案。

第七步第一版结果

data_generator_v2.py 已新增复杂缺陷：

1. rotated_rect
2. polygon
3. multi_defect

新增复杂 metadata 字段：

1. num_defects
2. component_types
3. component_centers
4. component_sizes
5. component_angles
6. polygon_vertices
7. num_vertices
8. min_mu
9. complexity_level

已生成小样本 v3 complex 数据集：

data/training_data_v3_complex_train.npz

data/training_data_v3_complex_val.npz

data/training_data_v3_complex_test.npz

小样本规模：

train = 20

val = 5

test = 5

验证结果

三个 npz 均包含 signals、mu_maps、defect_types、metadata、metadata_keys、x、y。

signals shape 和 mu_maps shape 正确。

train / val / test 中均出现 rotated_rect、polygon、multi_defect。

metadata 新字段和 metadata_keys 读取正常。

每个样本缺陷 mask 非空。

Bz signal 无 NaN / Inf。

可视化检查图已保存到：

results/previews/data_v3_complex_check_*.png

当前下一步

建议生成完整规模 v3 complex 数据集，但需等待用户确认。

建议命令：

& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" data_generator_v2.py --complex --train-samples 1000 --val-samples 200 --test-samples 200

执行约束

不要覆盖旧数据集：

data/training_data_train.npz

data/training_data_val.npz

data/training_data_test.npz

不要覆盖旧 checkpoints。

不要修改 train_pinn.py 的训练逻辑，除非明确进入复杂缺陷训练阶段。

不要修改 evaluate_pinn.py 的评价指标逻辑。

如果后续训练 v3 complex 模型，应使用新的 checkpoint 文件名，并将 simple baseline 和 complex baseline 分开记录。
