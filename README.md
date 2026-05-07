# PINN_project

## 项目简介

本项目是 PINN 缺陷边界形状反演 / 漏磁反演项目，目标是从一维 Bz 漏磁信号反演二维 μ map。

当前模型主线为：

`Bz signal + 空间坐标 (x, y) -> μ(x, y)`

其中 Bz signal 由 `BzEncoder` 编码为空间无关 latent vector，坐标使用 Fourier feature 编码，二者拼接后通过 MLP 输出二维 μ map。

## 当前项目进度

* [x] 第一步：`data_generator_v2.py` 批量数据 + metadata + train/val/test，已完成。
* [x] 第二步：`Bz signal + 坐标 -> μ map`，已完成。
* [x] 第三步：`evaluate_pinn.py` 定量评价，已完成。
* [x] 第四步：TV Loss，已完成。
* [x] 第五步：L-BFGS refine，流程已完成，但当前不作为默认推荐。
* [x] 第 5.5 步：TV / L-BFGS 参数扫描，已完成。
* [x] 第六步：物理一致性 Loss，已完成。
* [x] 第 6.5 步：物理一致性 Loss 效果验证与对比总结，已完成。
* [x] 第七步：复杂缺陷扩展第一版，已完成。
* [x] 第 7.5 步：v3 complex 复杂缺陷 baseline 训练，已完成。
* [x] 第 7.6 步：v3 complex 复杂缺陷 baseline 诊断分析，已完成。
* [x] 第 7.7 步：v3 complex 延长训练与专用 lambda_tv 扫描，已完成。

## 当前推荐 Baseline / 最佳模型

当前推荐模型：

`checkpoints/best_model_tv_5e-6.pt`

当前推荐配置：

* `lambda_tv = 5e-6`
* `lambda_phy = 0`，即默认不启用物理一致性 Loss 模型作为最佳模型
* L-BFGS：默认不启用，仅保留为 optional refine 实验

推荐依据：

* 第 5.5 步 TV 参数扫描中，`lambda_tv=5e-6` 按 `val_iou`、`val_dice`、`val_mae` 综合排名最佳。
* L-BFGS 小范围 refine 已跑通，但 val/test 指标整体不如最佳 TV 模型。
* 第六步物理一致性 Loss 初版已跑通，`lambda_phy=1e-4` 的模型为 `checkpoints/best_model_tv_phy.pt`；该模型 MSE 略有改善，但 MAE、IoU、Dice、area_error、center_error 均差于当前最佳 TV baseline。
* 第 6.5 步已完成正式对比总结，结果保存到 `results/summaries/physics_loss_comparison_summary.txt` 和 `results/metrics/physics_loss_comparison.csv`。结论是物理 Loss 初版不优于 baseline，不更新默认最佳模型。
* 第七步第一版只扩展了数据生成器，新增 `rotated_rect`、`polygon`、`multi_defect` 三类复杂缺陷。
* 第 7.5 步已经基于 v3 complex 数据集训练新的复杂缺陷 baseline：`checkpoints/best_model_v3_complex_tv.pt`。该模型和 simple baseline 对应的数据集不同，分开记录。
* 第 7.7 步完成 v3_complex 延长训练和专用 `lambda_tv` 扫描后，当前 v3_complex 推荐模型更新为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

当前推荐 baseline test 指标：

| 指标 | 数值 |
|---|---:|
| MSE | 2.16568206e+04 |
| MAE | 4.39399008e+01 |
| IoU | 4.32040206e-01 |
| Dice | 5.82132493e-01 |
| area_error | 2.42350201e-01 |
| center_error | 1.03291037e+00 |

当前 v3 complex 复杂缺陷 baseline：

* 模型路径：`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
* `lambda_tv = 2e-6`
* `lambda_phy = 0`
* L-BFGS：默认不启用
* 训练数据：`data/training_data_v3_complex_train.npz`
* 验证数据：`data/training_data_v3_complex_val.npz`
* 测试数据：`data/training_data_v3_complex_test.npz`

v3 complex baseline test 指标：

| 指标 | 数值 |
|---|---:|
| MSE | 2.07377174e+04 |
| MAE | 4.44655262e+01 |
| IoU | 2.95272047e-01 |
| Dice | 4.21885407e-01 |
| area_error | 3.94517442e-01 |
| center_error | 1.32594189e+00 |

第 7.6 步诊断结论：

* `polygon` 的 IoU / Dice 最低，IoU=2.20801408e-01，Dice=3.16226151e-01；
* `multi_defect` 的 MSE、MAE、center_error 更差，但 IoU / Dice 不是最低；
* 最差 10 个样本中 9 个是 `polygon`，说明复杂边界和小面积 polygon 漏检是当前主要问题之一；
* 诊断结果保存到 `results/summaries/v3_complex_diagnosis_summary.txt`。

第 7.7 步训练配置对比结论：

* 100 epoch 长训练模型 `checkpoints/best_model_v3_complex_tv_long.pt` 的 MSE 和 center_error 略有改善，但 MAE、IoU、Dice、area_error 变差，未作为默认 v3 complex 推荐模型。
* v3 complex 专用 `lambda_tv` 扫描按 val IoU / Dice / MAE 综合排序推荐 `lambda_tv=2e-6`。
* 推荐模型 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt` 相比 20 epoch v3 baseline 改善了整体 MSE、IoU、Dice、area_error、center_error，但 MAE 变差。
* polygon 的 IoU / Dice 有改善；multi_defect 的 MSE / MAE 和 mask 类指标仍是后续主要瓶颈之一。

第六步物理 Loss 初版模型：

* 模型路径：`checkpoints/best_model_tv_phy.pt`
* `lambda_phy = 1e-4`
* 结论：不作为当前默认最佳模型

第 6.5 步对比结论：

| 指标 | baseline `best_model_tv_5e-6.pt` | physics `best_model_tv_phy.pt` | 结论 |
|---|---:|---:|---|
| MSE | 2.16568206e+04 | 2.15898657e+04 | 改善 |
| MAE | 4.39399008e+01 | 4.45462792e+01 | 变差 |
| IoU | 4.32040206e-01 | 4.15690292e-01 | 变差 |
| Dice | 5.82132493e-01 | 5.65850626e-01 | 变差 |
| area_error | 2.42350201e-01 | 2.77560911e-01 | 变差 |
| center_error | 1.03291037e+00 | 1.03311558e+00 | 轻微变差 |

## 数据文件说明

数据集位于 `data/`：

* `data/training_data_train.npz`
* `data/training_data_val.npz`
* `data/training_data_test.npz`

每个 npz 文件包含：

* `signals`：一维 Bz 漏磁信号
* `mu_maps`：二维 μ map 标签
* `defect_types`：缺陷类型
* `metadata`：样本参数
* `metadata_keys`：metadata 字段说明
* `x`：x 坐标
* `y`：y 坐标

第七步已生成正式规模复杂缺陷数据集：

* `data/training_data_v3_complex_train.npz`
* `data/training_data_v3_complex_val.npz`
* `data/training_data_v3_complex_test.npz`

样本数量：

* train = 1000
* val = 200
* test = 200

v3 complex 数据集新增缺陷类型：

* `rotated_rect`
* `polygon`
* `multi_defect`

v3 complex metadata 在旧字段基础上新增：

* `num_defects`
* `component_types`
* `component_centers`
* `component_sizes`
* `component_angles`
* `polygon_vertices`
* `num_vertices`
* `min_mu`
* `complexity_level`

## 主要脚本说明

* `data_generator_v2.py`：批量生成 train / val / test 数据集，保存 signals、mu_maps、defect_types、metadata、metadata_keys、x、y；支持 `--complex` 生成 v3 complex 数据集。
* `train_pinn.py`：训练 BzEncoder + Fourier 坐标特征 + MLP 模型；支持 TV Loss、物理一致性 Loss 初版、L-BFGS refine；支持 `--dataset v3_complex` 自动读取复杂缺陷 train / val 数据集并使用新的输出路径。
* `evaluate_pinn.py`：加载测试集和 checkpoint，输出 MSE、MAE、IoU、Dice、area_error、center_error，并保存对比图；支持 `--dataset v3_complex` 和可选 metrics / figures 输出路径。
* `parameter_sweep.py`：执行 TV Loss 和 L-BFGS 参数扫描，生成参数对比表。

第六步没有新增独立脚本；物理一致性 Loss 加在 `train_pinn.py` 中。

## results 目录说明

`results/` 已按用途整理：

* `results/summaries/`：实验汇总说明，例如 `parameter_sweep_summary.txt`。
* `results/metrics/`：定量评价指标文件，例如 `*metrics*.txt`、`*metrics*.csv`。
* `results/loss_curves/`：训练损失曲线图，例如 `*loss_curve*.png`。
* `results/previews/`：预测预览、重建对比、评估样本图，例如 `*preview*.png`、`*reconstruction*.png`、`*prediction*.png`、`*evaluation_sample*.png`。
* `results/sweeps/`：参数扫描汇总表，例如 `tv_lambda_sweep.csv`、`lbfgs_sweep.csv`。
* `results/archive/`：训练日志、评估日志、`physics_loss_log.csv` 等辅助文件。

详细说明见 `results/README.md`。

第 6.5 步对比文件：

* `results/summaries/physics_loss_comparison_summary.txt`
* `results/metrics/physics_loss_comparison.csv`

第七步复杂缺陷可视化检查图：

* `results/previews/data_v3_complex_check_000.png`
* `results/previews/data_v3_complex_check_001.png`
* `results/previews/data_v3_complex_check_002.png`
* `results/previews/data_v3_complex_check_003.png`
* `results/previews/data_v3_complex_check_004.png`

正式规模 v3 complex 数据集检查摘要：

* `results/summaries/v3_complex_dataset_summary.txt`

第 7.5 步复杂缺陷 baseline 输出：

* `results/summaries/v3_complex_training_summary.txt`
* `results/metrics/evaluation_metrics_v3_complex_tv.csv`
* `results/metrics/evaluation_metrics_v3_complex_tv.txt`
* `results/loss_curves/loss_curve_v3_complex_tv.png`
* `results/previews/reconstruction_preview_v3_complex_tv.png`
* `results/previews/v3_complex_tv_evaluation_sample_000.png`
* `results/previews/v3_complex_tv_evaluation_sample_099.png`
* `results/previews/v3_complex_tv_evaluation_sample_199.png`

第 7.6 步复杂缺陷 baseline 诊断输出：

* `results/metrics/v3_complex_metrics_by_type.csv`
* `results/metrics/v3_complex_worst_samples.csv`
* `results/summaries/v3_complex_diagnosis_summary.txt`
* `results/previews/v3_complex_worst_samples/`

第 7.7 步复杂缺陷训练配置对比输出：

* `checkpoints/best_model_v3_complex_tv_long.pt`
* `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
* `results/summaries/v3_complex_long_training_summary.txt`
* `results/summaries/v3_complex_lambda_tv_sweep_summary.txt`
* `results/metrics/evaluation_metrics_v3_complex_tv_long.csv`
* `results/metrics/v3_complex_long_metrics_by_type.csv`
* `results/metrics/v3_complex_lambda_tv_sweep.csv`
* `results/metrics/evaluation_metrics_v3_complex_tv_sweep_2e-6_test.csv`
* `results/metrics/v3_complex_sweep_2e-6_test_metrics_by_type.csv`
* `results/loss_curves/loss_curve_v3_complex_tv_long.png`
* `results/loss_curves/loss_curve_v3_complex_tv_sweep_2e-6.png`
* `results/previews/reconstruction_preview_v3_complex_tv_long.png`
* `results/previews/reconstruction_preview_v3_complex_tv_sweep_2e-6.png`

## 常用运行命令

本项目脚本统一使用 Anaconda 环境 `pinn_mfl` 的解释器运行：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" 脚本名.py
```

生成数据集：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" data_generator_v2.py --train-samples 1000 --val-samples 200 --test-samples 200
```

生成 v3 complex 复杂缺陷数据集：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" data_generator_v2.py --complex --train-samples 1000 --val-samples 200 --test-samples 200
```

第七步第一版小样本验证命令：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" data_generator_v2.py --complex --train-samples 20 --val-samples 5 --test-samples 5
```

训练当前推荐 TV baseline：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" train_pinn.py --mode adam_tv --lambda-tv 5e-6 --checkpoint-path checkpoints/best_model_tv_5e-6.pt --loss-curve-path results/loss_curves/loss_curve_tv_5e-6.png --preview-path results/previews/reconstruction_preview_tv_5e-6.png
```

训练 v3 complex 复杂缺陷 baseline：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" train_pinn.py --mode adam_tv --dataset v3_complex --epochs 20 --lambda-tv 5e-6
```

训练当前推荐 v3 complex 模型：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" train_pinn.py --mode adam_tv --dataset v3_complex --epochs 50 --lambda-tv 2e-6 --checkpoint-path checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt --loss-curve-path results/loss_curves/loss_curve_v3_complex_tv_sweep_2e-6.png --preview-path results/previews/reconstruction_preview_v3_complex_tv_sweep_2e-6.png
```

训练第六步物理一致性 Loss 初版：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" train_pinn.py --mode adam_tv_phy --lambda-tv 5e-6 --lambda-phy 1e-4
```

评估当前推荐 baseline：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" evaluate_pinn.py --model checkpoints/best_model_tv_5e-6.pt --output_prefix best_tv
```

评估第六步物理 Loss 模型：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" evaluate_pinn.py --model checkpoints/best_model_tv_phy.pt --output_prefix tv_phy
```

评估 v3 complex 复杂缺陷 baseline：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" evaluate_pinn.py --dataset v3_complex --test-data data/training_data_v3_complex_test.npz --checkpoint checkpoints/best_model_v3_complex_tv.pt --output-prefix v3_complex_tv --metrics-csv results/metrics/evaluation_metrics_v3_complex_tv.csv --metrics-txt results/metrics/evaluation_metrics_v3_complex_tv.txt --figures-dir results/previews
```

评估当前推荐 v3 complex 模型：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" evaluate_pinn.py --dataset v3_complex --test-data data/training_data_v3_complex_test.npz --checkpoint checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt --output-prefix v3_complex_sweep_tv_2e-6_test --metrics-csv results/metrics/evaluation_metrics_v3_complex_tv_sweep_2e-6_test.csv --metrics-txt results/metrics/evaluation_metrics_v3_complex_tv_sweep_2e-6_test.txt --figures-dir results/previews
```

运行参数扫描：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" parameter_sweep.py --epochs 20
```

## 下一步计划

当前状态：第 7.7 步 v3 complex 延长训练与专用 lambda_tv 扫描已完成。

下一步等待用户确认。可选方向以 `NEXT_STEP.md` 为准：

1. 固定当前推荐 v3 complex 模型 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`；
2. 针对 polygon 和 multi_defect 继续做诊断；
3. 暂不进入 physics_loss 或 L-BFGS；
4. 模型结构优化可以作为后续方向，但不建议在没有进一步诊断前直接大改。

## 文档索引

* `PINN优化路线.md`：长期优化路线和阶段进度。
* `NEXT_STEP.md`：当前任务和下一步候选方向。
* `EXPERIMENT_LOG.md`：每次实验的参数、模型路径、指标和结论。
* `CURRENT_BASELINE.md`：当前推荐模型、推荐参数和 baseline 指标。
* `results/README.md`：results 子目录用途说明。
* `术语说明.md`：解释项目中常见术语和指标含义。
