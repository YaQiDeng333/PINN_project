# EXPERIMENT_LOG

## 第 5.5 步 TV Loss / L-BFGS 参数扫描

### 实验目的

选择进入第六步物理一致性 Loss 前的最佳 baseline。

### 修改文件

* parameter_sweep.py
* PINN优化路线.md
* NEXT_STEP.md

### 推荐配置

* lambda_tv = 5e-6
* 推荐模型 = checkpoints/best_model_tv_5e-6.pt
* L-BFGS 默认不启用，只保留为 optional refine

### 最佳 TV 模型 test 指标

* MSE = 2.1658206e+04
* MAE = 4.3939908e+01
* IoU = 4.3204206e-01
* Dice = 5.8213493e-01
* area_error = 2.4235021e-01
* center_error = 1.03291037e+00

### L-BFGS 结论

L-BFGS refine 流程已跑通，但当前小范围参数下 val/test 指标整体不如最佳 TV 模型，因此不作为默认推荐方案。

### 当前结论

第六步物理一致性 Loss 将以 checkpoints/best_model_tv_5e-6.pt 作为 baseline，lambda_tv = 5e-6，默认不启用 L-BFGS。

---

## 第六步：物理一致性 Loss 初版

### 实验目的

在当前最佳 baseline 基础上加入物理一致性 Loss，使预测 μ map 经过简化 forward model 后生成的 Bz_pred 与输入 Bz signal 匹配。

### 修改文件

* train_pinn.py
* PINN优化路线.md
* NEXT_STEP.md
* CURRENT_BASELINE.md
* EXPERIMENT_LOG.md

### baseline

* baseline 模型 = checkpoints/best_model_tv_5e-6.pt
* lambda_tv = 5e-6
* 默认不启用 L-BFGS

### physics_loss 定义

从预测 μ map 生成 soft defect mask，估计缺陷 soft area 和 center_x；
再使用 data_generator_v2.py 中信号生成形式的简化近似，根据 area、center_x、metadata 中的 depth 和 lift_off 生成 Bz_pred；
最后将 Bz_pred 按训练集 signal_mean / signal_std 归一化，与输入 normalized signals 做 MSE。

该 forward model 是简化版本：显式使用缺陷面积、center_x、depth、lift_off，不显式建模缺陷形状和 center_y。

### 训练配置

* lambda_phy = 1e-4
* total_loss = mse_loss + lambda_tv * tv_loss + lambda_phy * physics_loss
* 初始模型 = checkpoints/best_model_tv_5e-6.pt
* 输出模型 = checkpoints/best_model_tv_phy.pt
* 训练 epoch = 20

### 训练输出

* checkpoints/best_model_tv_phy.pt
* results/loss_curve_tv_phy.png
* results/reconstruction_preview_tv_phy.png
* results/physics_loss_log.csv

### physics_loss 训练量级

* 初始 physics_loss = 9.10055691e-02
* 最优 val_mse_loss 对应 epoch = 5
* epoch 5 physics_loss = 8.15256860e-02
* 最终 epoch 20 physics_loss = 7.74944600e-02

### test 指标

* MSE = 2.15898657e+04
* MAE = 4.45462792e+01
* IoU = 4.15690292e-01
* Dice = 5.65850626e-01
* area_error = 2.77560911e-01
* center_error = 1.03311558e+00

### 与 baseline 对比

baseline checkpoints/best_model_tv_5e-6.pt：

* MSE = 2.16568206e+04
* MAE = 4.39399008e+01
* IoU = 4.32040206e-01
* Dice = 5.82132493e-01
* area_error = 2.42350201e-01
* center_error = 1.03291037e+00

对比结论：

* MSE 改善；
* MAE 变差；
* IoU 变差；
* Dice 变差；
* area_error 变差；
* center_error 轻微变差。

### 当前结论

物理一致性 Loss 初版流程已跑通，但当前简化 forward model 和 lambda_phy = 1e-4 下，mask 类指标整体变差。

因此 checkpoints/best_model_tv_phy.pt 不作为默认最佳模型。

当前默认 baseline 仍保持 checkpoints/best_model_tv_5e-6.pt，lambda_tv = 5e-6，默认不启用 L-BFGS。

---

## 第 6.5 步：物理一致性 Loss 效果验证与对比总结

### 实验目的

判断第六步物理一致性 Loss 是否真正优于第 5.5 步 baseline。

### 修改文件

* results/summaries/physics_loss_comparison_summary.txt
* results/metrics/physics_loss_comparison.csv
* PINN优化路线.md
* NEXT_STEP.md
* README.md
* EXPERIMENT_LOG.md

### 对比模型

baseline 模型：

checkpoints/best_model_tv_5e-6.pt

物理 Loss 模型：

checkpoints/best_model_tv_phy.pt

### 对比结果

| 指标 | baseline | physics_loss | 结论 |
|---|---:|---:|---|
| MSE | 2.16568206e+04 | 2.15898657e+04 | 改善 |
| MAE | 4.39399008e+01 | 4.45462792e+01 | 变差 |
| IoU | 4.32040206e-01 | 4.15690292e-01 | 变差 |
| Dice | 5.82132493e-01 | 5.65850626e-01 | 变差 |
| area_error | 2.42350201e-01 | 2.77560911e-01 | 变差 |
| center_error | 1.03291037e+00 | 1.03311558e+00 | 轻微变差 |

### physics_loss / Bz reconstruction error

已有 physics_loss_log.csv 只记录了物理 Loss 训练过程。

物理 Loss 模型保存的最佳 val checkpoint 对应 epoch 5，该 epoch 的 train physics_loss = 8.15256860e-02。

baseline 模型没有记录同口径 physics_loss / Bz reconstruction error，本步骤未重新计算。

### 输出文件

* results/summaries/physics_loss_comparison_summary.txt
* results/metrics/physics_loss_comparison.csv

### 当前结论

第六步物理 Loss 模型只在 MSE 上略优，mask 类指标 IoU、Dice、area_error、center_error 均变差，MAE 也变差。

因此 checkpoints/best_model_tv_phy.pt 不作为默认最佳模型。

当前推荐模型仍为 checkpoints/best_model_tv_5e-6.pt，lambda_tv = 5e-6，lambda_phy = 0，默认不启用 L-BFGS。

物理一致性 Loss 初版建议保留为实验模块；如后续继续优化，应先扫描 lambda_phy 或改进 forward model。

完成第 6.5 步后，可以在用户确认后进入第七步复杂缺陷扩展。

---

## 第七步：复杂缺陷扩展第一版

### 实验目的

在 data_generator_v2.py 中扩展更复杂、更接近真实情况的缺陷类型，同时保持现有训练和评价逻辑不变。

### 修改文件

* data_generator_v2.py
* README.md
* PINN优化路线.md
* NEXT_STEP.md
* EXPERIMENT_LOG.md

### 新增缺陷类型

* rotated_rect：旋转矩形；
* polygon：不规则多边形；
* multi_defect：同一 μ map 中多个缺陷。

### 新增 metadata 字段

* num_defects
* component_types
* component_centers
* component_sizes
* component_angles
* polygon_vertices
* num_vertices
* min_mu
* complexity_level

metadata_keys 已同步更新。

### 小样本生成命令

& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" data_generator_v2.py --complex --train-samples 20 --val-samples 5 --test-samples 5

### 生成的数据文件

* data/training_data_v3_complex_train.npz
* data/training_data_v3_complex_val.npz
* data/training_data_v3_complex_test.npz

旧数据集未覆盖：

* data/training_data_train.npz
* data/training_data_val.npz
* data/training_data_test.npz

### 验证结果

* npz 文件能正常生成；
* 每个 npz 均包含 signals、mu_maps、defect_types、metadata、metadata_keys、x、y；
* train signals shape = (20, 200)，mu_maps shape = (20, 100, 200)；
* val signals shape = (5, 200)，mu_maps shape = (5, 100, 200)；
* test signals shape = (5, 200)，mu_maps shape = (5, 100, 200)；
* train / val / test 中均出现 rotated_rect、polygon、multi_defect；
* metadata 新字段可正常读取；
* 每个样本缺陷 mask 非空；
* Bz signal 无 NaN / Inf。

### 可视化检查图

已保存 5 张检查图：

* results/previews/data_v3_complex_check_000.png
* results/previews/data_v3_complex_check_001.png
* results/previews/data_v3_complex_check_002.png
* results/previews/data_v3_complex_check_003.png
* results/previews/data_v3_complex_check_004.png

### 当前结论

复杂缺陷数据生成第一版已跑通。

当前未重新训练模型，未修改 train_pinn.py，未修改 evaluate_pinn.py，未覆盖旧 checkpoint。

当前推荐模型仍为 checkpoints/best_model_tv_5e-6.pt。

### 下一步建议

建议下一步生成完整规模 v3 complex 数据集：

train = 1000

val = 200

test = 200

随后再单独训练复杂缺陷模型，使用新的 checkpoint 文件名，不覆盖当前 baseline。

---

## 历史补录：第一步 data_generator 数据集生成

### 目标

让 data_generator 支持批量生成样本，并保存完整 metadata。

### 主要结果

* 已生成 train / val / test 三个数据集；
* train = 1000 个样本；
* val = 200 个样本；
* test = 200 个样本；
* 每个 npz 包含 signals、mu_maps、defect_types、metadata、x、y；
* 数据文件：

  * data/training_data_train.npz
  * data/training_data_val.npz
  * data/training_data_test.npz

### 结论

第一步完成，后续训练不再只依赖单个样本。

---

## 历史补录：第二步 train_pinn 输入 Bz 信号

### 目标

将模型从“只输入坐标”升级为“Bz signal + 坐标 → μ map”。

### 主要结果

* train_pinn.py 已加入 BzEncoder；
* 使用 signals 作为 Bz 输入；
* 使用 mu_maps 作为监督标签；
* 支持 train / val 数据集；
* 保存 best_model.pt；
* 生成 loss 曲线和预测对比图；
* 具体数值指标：当时未单独记录，待补充。

### 结论

第二步完成，模型开始具备由漏磁信号反演 μ map 的基本结构。

---

## 历史补录：第三步 evaluate_pinn 定量评价

### 目标

建立测试集评价流程，加入定量指标。

### 主要结果

* 新增或修改 evaluate_pinn.py；
* 支持加载 checkpoints/best_model.pt；
* 支持评估 data/training_data_test.npz；
* 输出指标：

  * MSE
  * MAE
  * IoU
  * Dice
  * area_error
  * center_error

* 保存 evaluation_metrics.txt 和 evaluation_metrics.csv。

### test 指标

* MSE = 2.17269746e+04
* MAE = 4.61076419e+01
* IoU = 4.25961039e-01
* Dice = 5.76173878e-01
* area_error = 2.69236126e-01
* center_error = 1.02991446e+00

### 结论

第三步完成，后续可以用统一指标比较不同模型。

---

## 历史补录：第四步 TV Loss

### 目标

在 MSE Loss 基础上加入 TV Loss，减少 μ map 毛刺和孤立噪点。

### 主要结果

* train_pinn.py 已加入 tv_loss；
* 损失形式：

  total_loss = mse_loss + lambda_tv * tv_loss

* 初始 lambda_tv = 1e-4；
* 生成 best_model_tv.pt；
* 生成 loss_curve_tv.png 和 reconstruction_preview_tv.png。

### test 指标

* MSE = 2.17338678e+04
* MAE = 4.29366580e+01
* IoU = 4.11922591e-01
* Dice = 5.60657765e-01
* area_error = 2.69431885e-01
* center_error = 1.04590862e+00

### 实验结论

TV Loss 流程已跑通。当前 lambda_tv = 1e-4 下，MAE 有改善，但 IoU、Dice、area_error、center_error 等 mask 类指标没有明显改善，因此需要进一步扫描 lambda_tv。

---

## 历史补录：第五步 L-BFGS refine

### 目标

在 TV Loss 模型基础上加入 L-BFGS 后期精修。

### 主要结果

* L-BFGS refine 流程已独立跑通；
* 从 best_model_tv.pt 加载模型；
* 没有覆盖 best_model.pt 或 best_model_tv.pt；
* 生成 best_model_tv_lbfgs.pt；
* L-BFGS 当前作为 optional refine。

### test 指标

* MSE = 2.62752306e+04
* MAE = 4.48119284e+01
* IoU = 3.49731439e-01
* Dice = 4.86349610e-01
* area_error = 4.15055531e-01
* center_error = 1.15353433e+00

### 实验结论

当前小子集 refine 设置下，泛化指标整体变差。因此 L-BFGS 暂不作为默认推荐方案，只保留为可选实验。

---

## 历史补录：第 5.5 步 TV Loss / L-BFGS 参数扫描

### 目标

在进入第六步物理一致性 Loss 前，选择当前最优 baseline。

### 主要结果

* 新增 parameter_sweep.py；
* 完成 TV lambda 扫描；
* 完成 L-BFGS 小范围参数扫描；
* 保存：

  * results/tv_lambda_sweep.csv
  * results/lbfgs_sweep.csv
  * results/parameter_sweep_summary.txt

### 推荐配置

* 推荐 lambda_tv = 5e-6
* 推荐模型 = checkpoints/best_model_tv_5e-6.pt
* L-BFGS 默认不启用，只保留为 optional refine

### 最佳 TV 模型 test 指标

* MSE = 2.16568206e+04
* MAE = 4.39399008e+01
* IoU = 4.32040206e-01
* Dice = 5.82132493e-01
* area_error = 2.42350201e-01
* center_error = 1.03291037e+00

### 最佳 L-BFGS 候选 test 指标

* MSE = 2.22554668e+04
* MAE = 5.02426600e+01
* IoU = 4.22638392e-01
* Dice = 5.73455660e-01
* area_error = 2.69747073e-01
* center_error = 1.04688911e+00

### 结论

第六步物理一致性 Loss 将以 checkpoints/best_model_tv_5e-6.pt 作为 baseline，lambda_tv = 5e-6，默认不启用 L-BFGS。
