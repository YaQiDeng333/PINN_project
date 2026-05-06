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
* 第七步第一版只扩展了数据生成器，新增 `rotated_rect`、`polygon`、`multi_defect` 三类复杂缺陷；当前未重新训练模型，因此默认最佳模型仍不变。

当前推荐 baseline test 指标：

| 指标 | 数值 |
|---|---:|
| MSE | 2.16568206e+04 |
| MAE | 4.39399008e+01 |
| IoU | 4.32040206e-01 |
| Dice | 5.82132493e-01 |
| area_error | 2.42350201e-01 |
| center_error | 1.03291037e+00 |

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

第七步第一版新增复杂缺陷小样本数据集：

* `data/training_data_v3_complex_train.npz`
* `data/training_data_v3_complex_val.npz`
* `data/training_data_v3_complex_test.npz`

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
* `train_pinn.py`：训练 BzEncoder + Fourier 坐标特征 + MLP 模型；支持 TV Loss、物理一致性 Loss 初版、L-BFGS refine。
* `evaluate_pinn.py`：加载测试集和 checkpoint，输出 MSE、MAE、IoU、Dice、area_error、center_error，并保存对比图。
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

运行参数扫描：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" parameter_sweep.py --epochs 20
```

## 下一步计划

当前状态：第七步复杂缺陷扩展第一版已完成，默认最佳 baseline 仍为 `checkpoints/best_model_tv_5e-6.pt`。

下一步等待用户确认。可选方向以 `NEXT_STEP.md` 为准：

1. 生成完整规模 v3 complex 数据集；
2. 基于 v3 complex train / val 训练新的复杂缺陷模型；
3. 继续改进复杂缺陷生成质量或 metadata 表达。

## 文档索引

* `PINN优化路线.md`：长期优化路线和阶段进度。
* `NEXT_STEP.md`：当前任务和下一步候选方向。
* `EXPERIMENT_LOG.md`：每次实验的参数、模型路径、指标和结论。
* `CURRENT_BASELINE.md`：当前推荐模型、推荐参数和 baseline 指标。
* `results/README.md`：results 子目录用途说明。
* `术语说明.md`：解释项目中常见术语和指标含义。
