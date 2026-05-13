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
* [x] 第 7.8 步：polygon / multi_defect 细诊断，已完成。
* [x] 第 7.9 步：v4 balanced complex 数据增强与样本平衡及正式规模数据集生成，已完成。
* [x] 第 7.10 步：v4 balanced complex baseline 训练，已完成。
* [x] 第 7.11 步：v4 balanced complex 专用 lambda_tv 扫描，已完成。

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

第 7.8 步细诊断结论：

* polygon 按顶点数分组后不是“顶点数越多越差”：5 顶点 polygon 的 IoU / Dice 最低，8 和 9 顶点反而更好。
* polygon 最差 10 个样本全部 IoU=0、Dice=0、pred_area=0，主要问题是预测 mask 为空或漏检，而不是单纯边数更多。
* multi_defect 从 2 个缺陷到 3 个缺陷时，MSE、MAE、area_error、center_error 明显变差；center_error 从 1.34234157e+00 升到 2.22782234e+00。
* complexity_level=3 相比 level=2 的 MSE、MAE、area_error、center_error 变差，但 IoU/Dice 没有同步下降，说明复杂度影响主要体现在数值误差和中心偏移。

第 7.9 步数据增强实现状态：

* `data_generator_v2.py` 新增 `--dataset v4_balanced_complex`；
* v4 正式规模数据集已生成并通过检查：train=1000、val=200、test=200；
* 生成 seed = 7904；
* v4 数据增强重点为：5 顶点 polygon、小面积/弱信号 polygon 过滤、2/3 缺陷 multi_defect 平衡、complexity_level 近似平衡；
* polygon `area_bin` 当前阈值为：small < 120 pixels，120 <= medium < 500 pixels，large >= 500 pixels；
* polygon 生成已取消双层 retry，当前为样本级单层有限重试，同时检查 `mask_pixels >= 30` 和 `signal_snr >= 5`；
* 当前未重新训练模型，当前 v3 complex 推荐模型仍为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

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

第 7.9 步已生成正式规模 v4 balanced complex 数据集：

* `data/training_data_v4_balanced_complex_train.npz`
* `data/training_data_v4_balanced_complex_val.npz`
* `data/training_data_v4_balanced_complex_test.npz`

样本数量：

* train = 1000
* val = 200
* test = 200

v4 metadata 在 v3 字段基础上新增：

* `mask_pixels`
* `signal_peak_to_peak`
* `signal_snr`
* `area_bin`
* `balance_group`

说明：由于 metadata dtype 在 `data_generator_v2.py` 中是全局定义，如果以后重新生成 simple 或 v3_complex 数据集，新 npz 也可能自动包含这些 v4 新增 metadata 字段。旧数据集文件本身没有被覆盖。

## 主要脚本说明

* `data_generator_v2.py`：批量生成 train / val / test 数据集，保存 signals、mu_maps、defect_types、metadata、metadata_keys、x、y；支持 `--complex` 生成 v3 complex 数据集，支持 `--dataset v4_balanced_complex` 生成 v4 平衡复杂缺陷数据集。
* `train_pinn.py`：训练 BzEncoder + Fourier 坐标特征 + MLP 模型；支持 TV Loss、物理一致性 Loss 初版、L-BFGS refine；支持 `--dataset v3_complex` 和 `--dataset v4_balanced_complex` 自动读取对应 train / val 数据集并使用新的输出路径。
* `evaluate_pinn.py`：加载测试集和 checkpoint，输出 MSE、MAE、IoU、Dice、area_error、center_error，并保存对比图；支持 `--dataset v3_complex`、`--dataset v4_balanced_complex` 和可选 metrics / figures 输出路径。
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

第 7.8 步 polygon / multi_defect 细诊断输出：

* `results/metrics/v3_complex_polygon_by_vertices.csv`
* `results/metrics/v3_complex_multi_defect_by_count.csv`
* `results/metrics/v3_complex_by_complexity_level.csv`
* `results/metrics/v3_complex_polygon_worst10.csv`
* `results/metrics/v3_complex_multi_defect_worst10.csv`
* `results/summaries/v3_complex_fine_diagnosis_summary.txt`
* `results/previews/v3_complex_fine_diagnosis/`

第 7.9 步 v4 balanced complex 正式数据集输出：

* `results/summaries/v4_balanced_complex_dataset_summary.txt`

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

生成 v4 balanced complex 正式规模数据集：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" data_generator_v2.py --dataset v4_balanced_complex --train-samples 1000 --val-samples 200 --test-samples 200 --seed 7904 --visual-check-samples 0
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

当前状态：第 7.9 步 v4 balanced complex 正式规模数据集生成已完成。

下一步等待用户确认。可选方向以 `NEXT_STEP.md` 为准：

1. 下一步可以进入 v4_balanced_complex 模型训练；
2. 暂不进入 physics_loss、L-BFGS 或模型结构大改；
3. v4 训练应保存新 checkpoint，不覆盖 v3_complex 推荐模型。

## 文档索引

* `PINN优化路线.md`：长期优化路线和阶段进度。
* `NEXT_STEP.md`：当前任务和下一步候选方向。
* `EXPERIMENT_LOG.md`：每次实验的参数、模型路径、指标和结论。
* `CURRENT_BASELINE.md`：当前推荐模型、推荐参数和 baseline 指标。
* `results/README.md`：results 子目录用途说明。
* `术语说明.md`：解释项目中常见术语和指标含义。

---

## 第 7.10 步补充：v4_balanced_complex baseline 训练

当前已完成 v4_balanced_complex 正式数据集 baseline 训练和评估。

主要输出：

* checkpoints/best_model_v4_balanced_complex_tv.pt
* results/loss_curves/loss_curve_v4_balanced_complex_tv.png
* results/previews/reconstruction_preview_v4_balanced_complex_tv.png
* results/metrics/evaluation_metrics_v4_balanced_complex_tv.csv
* results/metrics/evaluation_metrics_v4_balanced_complex_tv.txt
* results/metrics/v4_balanced_complex_metrics_by_type.csv
* results/metrics/v4_balanced_complex_polygon_by_area_bin.csv
* results/metrics/v4_balanced_complex_polygon_by_vertices.csv
* results/metrics/v4_balanced_complex_multi_defect_by_count.csv
* results/metrics/v4_balanced_complex_by_complexity_level.csv
* results/summaries/v4_balanced_complex_training_summary.txt
* results/summaries/v4_balanced_complex_diagnosis_summary.txt

v4 test 整体指标：MSE=2.39571663e+04，MAE=4.88803274e+01，IoU=2.67902294e-01，Dice=3.81393009e-01，area_error=4.79983772e-01，center_error=1.41093149e+00。

当前结论：v4 数据更平衡，但本轮 baseline 未明显超过 v3_complex 推荐模型 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt，因此当前推荐 baseline 不变。下一步以 NEXT_STEP.md 为准，优先做 v4 专属 lambda_tv 扫描。

常用命令：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" train_pinn.py --mode adam_tv --dataset v4_balanced_complex --epochs 100 --lambda-tv 2e-6
```

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" evaluate_pinn.py --dataset v4_balanced_complex --test-data data/training_data_v4_balanced_complex_test.npz --checkpoint checkpoints/best_model_v4_balanced_complex_tv.pt --output-prefix v4_balanced_complex_tv --metrics-csv results/metrics/evaluation_metrics_v4_balanced_complex_tv.csv --metrics-txt results/metrics/evaluation_metrics_v4_balanced_complex_tv.txt --figures-dir results/previews
```

## 第 7.11 步补充：v4_balanced_complex lambda_tv 扫描

已完成 v4_balanced_complex 专用 `lambda_tv` 扫描：

* 候选值：0、5e-7、1e-6、2e-6、5e-6、1e-5
* 每组训练：50 epoch
* physics_loss：未启用
* L-BFGS：未启用
* 模型结构：未修改
* 评价指标定义：未修改

主要输出：

* `results/metrics/v4_balanced_complex_lambda_tv_sweep.csv`
* `results/summaries/v4_balanced_complex_lambda_tv_sweep_summary.txt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_0.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_5e-7.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_1e-6.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_2e-6.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_5e-6.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_1e-5.pt`

按 val IoU、Dice、MAE、area_error、center_error 综合排序，本轮推荐候选为 `lambda_tv=0`，模型为 `checkpoints/best_model_v4_balanced_complex_tv_sweep_0.pt`。该候选 test 指标为：MSE=2.41644578e+04，MAE=5.09550103e+01，IoU=2.73743067e-01，Dice=3.87241381e-01，area_error=4.90251054e-01，center_error=1.38652205e+00。

当前结论：v4 sweep 没有解决 small polygon 漏检问题，small polygon IoU/Dice 仍为 0；v4 sweep 候选仍未明显优于当前 v3_complex 推荐模型 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`，因此不切换 `CURRENT_BASELINE.md`。下一步建议进入模型结构或训练策略优化，而不是继续扩大 `lambda_tv` 扫描。

## 第 7.12A 步补充：small polygon defect-weighted MSE Loss

已完成 v4_balanced_complex 上的 defect-weighted MSE 实验。`train_pinn.py` 新增 `--loss-type mse / weighted_mse`，默认仍为 `mse`，因此旧训练流程默认不受影响；新增 `--defect-weight`，本轮使用 `10.0`。

本轮训练配置：

* 数据集：v4_balanced_complex
* loss_type = weighted_mse
* defect_weight = 10.0
* lambda_tv = 0
* epoch = 100
* physics_loss / L-BFGS / soft Dice / oversampling：均未启用

主要输出：

* `checkpoints/best_model_v4_balanced_complex_smallpoly_loss.pt`
* `results/loss_curves/loss_curve_v4_smallpoly_loss.png`
* `results/previews/reconstruction_preview_v4_smallpoly_loss.png`
* `results/metrics/evaluation_metrics_v4_smallpoly_loss.csv`
* `results/metrics/evaluation_metrics_v4_smallpoly_loss.txt`
* `results/summaries/v4_smallpoly_loss_summary.txt`

v4 test 整体指标：MSE=4.10216735e+04，MAE=7.83255570e+01，IoU=3.22104979e-01，Dice=4.67866207e-01，area_error=1.34222578e+00，center_error=1.14444251e+00。

small polygon 已不再全部漏检：IoU=1.36334593e-01，Dice=2.26148223e-01，`pred_area=0` 的样本数为 0/25。代价是 MSE、MAE、area_error 明显变差，说明 `defect_weight=10.0` 可能导致预测缺陷区域偏大。当前不切换 `CURRENT_BASELINE.md`，v3_complex 推荐模型仍为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

下一步建议进行 defect_weight 小范围扫描，候选值为 5 / 10 / 20 / 50，暂不加入 soft Dice、oversampling、physics_loss、L-BFGS 或模型结构改动。

## 第 7.12B 步补充：v4 small polygon defect_weight 扫描

已完成 v4_balanced_complex 上的 `defect_weight` 扫描。训练仍使用 `weighted_mse`，`lambda_tv=0`，不启用 physics_loss、L-BFGS、Dice Loss 或 oversampling，不修改模型结构和评价指标定义。

候选值：

* 2
* 3
* 5
* 7
* 10

主要输出：

* `results/metrics/v4_smallpoly_defect_weight_sweep.csv`
* `results/summaries/v4_smallpoly_defect_weight_sweep_summary.txt`
* `checkpoints/best_model_v4_smallpoly_w2.pt`
* `checkpoints/best_model_v4_smallpoly_w3.pt`
* `checkpoints/best_model_v4_smallpoly_w5.pt`
* `checkpoints/best_model_v4_smallpoly_w7.pt`
* `checkpoints/best_model_v4_smallpoly_w10.pt`

当前 v4 small polygon weighted MSE 推荐候选为 `defect_weight=5`：

* 模型：`checkpoints/best_model_v4_smallpoly_w5.pt`
* MSE = 3.12321945e+04
* MAE = 6.23678583e+01
* IoU = 3.39080635e-01
* Dice = 4.77603301e-01
* area_error = 8.38023859e-01
* center_error = 1.17307553e+00
* small polygon IoU = 6.54854895e-02
* small polygon Dice = 1.04442883e-01
* small polygon pred_area=0：12 / 25

结论：`defect_weight=5` 相比 `defect_weight=10` 明显降低 area_error，并保持 small polygon 有效重叠检出；但该模型尚不足以替代当前 v3_complex 推荐 baseline，因此 `CURRENT_BASELINE.md` 不切换。下一步建议先做 Claude Code review，再考虑是否进入 soft Dice / focal 类 loss。

## 第 7.13 步补充：weighted MSE + soft Dice Loss

已完成 v4_balanced_complex 上的 weighted MSE + soft Dice Loss 实验。训练仍不修改模型结构，不启用 physics_loss、L-BFGS 或 oversampling，也不修改 `evaluate_pinn.py` 的评价指标定义。

训练配置：

* loss_type = weighted_mse_dice
* defect_weight = 5
* lambda_dice = 0.05
* lambda_tv = 0
* epoch = 100

主要输出：

* `checkpoints/best_model_v4_smallpoly_w5_dice.pt`
* `results/loss_curves/loss_curve_v4_smallpoly_w5_dice.png`
* `results/previews/reconstruction_preview_v4_smallpoly_w5_dice.png`
* `results/metrics/evaluation_metrics_v4_smallpoly_w5_dice.csv`
* `results/metrics/evaluation_metrics_v4_smallpoly_w5_dice.txt`
* `results/summaries/v4_smallpoly_w5_dice_summary.txt`

v4 test 整体指标：MSE=3.56734905e+04，MAE=6.02042826e+01，IoU=3.25826098e-01，Dice=4.64347405e-01，area_error=6.12110696e-01，center_error=1.24440727e+00。

与第 7.12B 的 `defect_weight=5` weighted MSE 相比，soft Dice 将 small polygon `pred_area=0` 从 12/25 降到 0/25，small polygon IoU/Dice 提升到 1.26014768e-01 / 2.01116176e-01，area_error 也降低；但 overall IoU/Dice 下降，multi_defect center_error 变差。当前不切换 `CURRENT_BASELINE.md`。

下一步建议做 `lambda_dice` 小范围扫描，而不是直接切换 baseline。

---

## 第 7.13B 步补充：v4 small polygon lambda_dice 扫描

已完成 v4_balanced_complex 上的 `lambda_dice` 扫描，训练配置为 `loss_type=weighted_mse_dice`、`defect_weight=5`、`lambda_tv=0`、100 epoch，不启用 physics_loss、L-BFGS 或 oversampling。

主要输出：
* `results/metrics/v4_smallpoly_lambda_dice_sweep.csv`
* `results/summaries/v4_smallpoly_lambda_dice_sweep_summary.txt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt`

本轮推荐 v4 small polygon 专项候选为 `checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt`，对应 `lambda_dice=0.03`。该候选让 small polygon `pred_area=0` 降为 0/25，且 overall IoU / Dice 和 multi_defect center_error 相比 weighted MSE w5 有改善；但 area_error 仍偏大，因此不替换当前全项目 baseline。

当前全项目推荐 baseline 仍以 `CURRENT_BASELINE.md` 为准：`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。
---

## 第 7.14 步补充：area_error 诊断

已完成 `checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt` 的 area_error 诊断。诊断结果保存到：

* `results/metrics/v4_smallpoly_area_error_per_sample.csv`
* `results/metrics/v4_smallpoly_area_error_by_type.csv`
* `results/metrics/v4_smallpoly_area_error_by_area_bin.csv`
* `results/metrics/v4_smallpoly_area_error_by_vertices.csv`
* `results/metrics/v4_smallpoly_area_error_by_num_defects.csv`
* `results/metrics/v4_smallpoly_area_error_by_complexity.csv`
* `results/metrics/v4_smallpoly_area_error_worst10.csv`
* `results/summaries/v4_smallpoly_area_error_diagnosis_summary.txt`
* `results/previews/v4_smallpoly_area_error_worst10/`

结论：`lambda_dice=0.03` 的面积误差主要来自系统性面积高估，200 个 test 样本中 190 个 `pred_area > true_area`。最严重问题集中在 polygon，尤其 small / medium polygon；multi_defect 不是本轮 area_error 主因。当前不切换全项目 baseline，推荐 baseline 仍以 `CURRENT_BASELINE.md` 为准。
---

## 第 7.15 步补充：area-aware loss 面积约束实验

已完成 v4_balanced_complex 上的 area-aware loss 扫描。`train_pinn.py` 新增可选 `weighted_mse_dice_area` loss 和 `--lambda-area` 参数，默认 `--loss-type` 仍为 `mse`，旧训练流程默认不受影响。

主要输出：

* `results/metrics/v4_smallpoly_area_loss_sweep.csv`
* `results/summaries/v4_smallpoly_area_loss_sweep_summary.txt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_0p05.pt`

本轮推荐 v4 area-aware 专项候选为 `checkpoints/best_model_v4_smallpoly_w5_dice_area_0p05.pt`，对应 `lambda_area=0.05`。它能明显降低 overall area_error、polygon area_error 和 medium polygon area_error，并保持 small polygon `pred_area=0` 为 0/25；但 overall IoU / Dice 略降，multi_defect center_error 略变差，因此不替换当前全项目 baseline。

当前全项目推荐 baseline 仍以 `CURRENT_BASELINE.md` 为准：`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。
---

## 第 7.16 步补充：面积约束细化实验

已完成 v4_balanced_complex 上的面积约束细化实验。`train_pinn.py` 新增 `--area-loss-type symmetric / over_only`，默认仍为 `symmetric`。loss curve 标题已改为通用的 `Training Loss Curve`。

主要输出：
* `results/metrics/v4_smallpoly_area_loss_refine.csv`
* `results/summaries/v4_smallpoly_area_loss_refine_summary.txt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_refine_0p07.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_over_0p05.pt`

结论：symmetric `lambda_area=0.07` 是面积指标最好的 symmetric 候选；`over_only` 虽然更能降低 `pred_area > true_area` 数量，但会让 small polygon 漏检回升，因此不作为当前推荐方案。当前全项目 baseline 不变。
---

## 最新实验状态：第 7.17 步

第 7.17 步已完成 v4_balanced_complex 上的 symmetric area loss 组合验证。

本轮固定：

* `loss_type = weighted_mse_dice_area`
* `lambda_dice = 0.03`
* `lambda_tv = 0`
* `area_loss_type = symmetric`
* `epochs = 100`

验证组合为 `defect_weight = 5 / 7` 与 `lambda_area = 0.04 / 0.07`。本轮综合表现最好的 v4 small polygon / area loss 候选为：

`checkpoints/best_model_v4_w5_dice003_area004.pt`

对应 `defect_weight=5, lambda_area=0.04`。该模型保持 small polygon `pred_area=0` 为 0 / 25，并在本轮四组中取得最高 overall IoU / Dice 和最低 multi_defect center_error；但 polygon area_error 未继续下降，不如第 7.16 步的 symmetric `lambda_area=0.07`。

因此当前不切换全项目 baseline。当前推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

相关输出：

* `results/metrics/v4_smallpoly_weight_area_combo.csv`
* `results/summaries/v4_smallpoly_weight_area_combo_summary.txt`
---

## 最新实验状态：第 7.18 步

第 7.18 步已完成后处理与阈值分析，分析对象为第 7.17 步推荐的 v4 small polygon / area-aware 候选：

`checkpoints/best_model_v4_w5_dice003_area004.pt`

本轮不重新训练，不修改模型结构，也不修改 `evaluate_pinn.py` 中现有标准指标定义。

主要结论：

* 标准 mask threshold=500 时，area_error = 0.911511，pred_area > true_area = 191 / 200；
* threshold=300 时，area_error 降至 0.292975，pred_area > true_area 降至 114 / 200；
* threshold=300 时，small polygon `pred_area=0` 仍为 0 / 25；
* threshold=450 的 IoU / Dice 更高，分别为 0.354303 / 0.497498；
* 连通域过滤 remove < 5 / 10 / 20 pixels 基本没有额外收益；
* 后处理可作为可选评估方案，但不替代标准 `evaluate_pinn.py` 流程。

相关输出：

* `results/metrics/v4_postprocess_threshold_sweep.csv`
* `results/metrics/v4_postprocess_component_filter.csv`
* `results/summaries/v4_postprocess_analysis_summary.txt`
* `results/previews/v4_postprocess_examples/`

当前全项目推荐 baseline 不变：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
---

## 最新实验状态：第 7.18.5 步

第 7.18.5 步已完成训练随机种子支持，为第 7.19 模型结构优化实验做准备。

`train_pinn.py` 新增：

* `--seed` 参数；
* 默认值 `42`；
* `set_seed(seed)`，同步设置 Python random、NumPy、PyTorch 和 CUDA 随机种子；
* Adam 训练中 `DataLoader(shuffle=True)` 使用固定 `torch.Generator()`；
* 训练启动时打印当前 seed。

从本步开始，后续训练默认固定 `seed=42`。第 7.19 及之后的结构对比实验必须固定 seed，关键结论建议做 repeat 验证。

第 7.18 的后处理阈值分析还说明，当前模型预测 μ 值存在校准偏软问题：缺陷区域常预测为 μ≈200–400，而不是接近真实 μ≈1。因此 threshold=300 能显著降低 area_error。这是模型输出校准问题，不是单纯评估阈值问题。

常用示例：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" train_pinn.py --dataset v4_balanced_complex --seed 42
```

## 第 7.19 步：模型结构优化方案

第 7.19 步已完成方案设计，未修改训练代码，未重新训练模型。新增方案文档：

* `MODEL_STRUCTURE_PLAN.md`：记录当前结构分析、输出 μ 偏软问题定位、推荐结构优化方案和第 7.20A / 7.20B 实施计划。

当前关键判断：

* 第 7.12-7.18 的 loss 和后处理实验已经缓解 small polygon 漏检；
* 但第 7.18 后处理阈值分析显示，缺陷区域预测 μ 值常停留在 `μ_r≈200-400`，没有接近真实缺陷 `μ_r≈1`；
* 当前输出层实际是 `Linear + Softplus`。Softplus 有下界但无上界，缺陷端要逼近 `mu_norm≈0.001` 时需要很负的 pre-activation，可能导致输出偏软；
* threshold=300 能显著降低 area_error，说明这是模型输出校准和边界表达问题，不应只靠修改评价阈值解决；
* 下一步建议进入第 7.20A：输出 μ 参数化校准实验。

第 7.20A 推荐方向：

* 保留 BzEncoder 和 Fourier feature；
* 新增可选 `calibrated_mu` 模型变体；
* 让 decoder 先预测 defect probability，再映射到归一化 μ 范围 `[0.001, 1.0]`；
* 保持当前 decoder 结构不变，即 `128 / 128 / 64 + Tanh`；
* 固定 `seed=42` 做旧结构和新结构 A/B 对比；
* 不启用 physics_loss，不启用 L-BFGS，不切换 CURRENT_BASELINE。

第 7.20B 暂不立即执行。只有当第 7.20A 有效或部分有效后，再考虑增强 decoder，例如 `256 / 256 / 128 / 64 + SiLU`。

文档索引补充：

* `MODEL_STRUCTURE_PLAN.md`：第 7.19 模型结构优化方案和第 7.20A / 7.20B 实施计划。

---

## 最新实验状态：第 7.20A 步

第 7.20A 步已完成 `calibrated_mu` 输出 μ 参数化校准实验。

本轮只修改输出 μ 参数化，不增强 BzEncoder，不增强 decoder，不启用 physics_loss / L-BFGS，也不修改标准评价指标定义。`train_pinn.py` 新增：

* `--model-variant baseline / calibrated_mu`
* 默认 `baseline`
* `baseline` 保持旧输出行为：`Linear(64, 1) + Softplus`
* `calibrated_mu` 将 decoder logit 映射为 defect probability，再映射到 `mu_norm ∈ [0.001, 1.0]`

固定训练配置：

* dataset = `v4_balanced_complex`
* seed = 42
* loss_type = `weighted_mse_dice_area`
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.04
* area_loss_type = `symmetric`
* lambda_tv = 0
* epochs = 100

主要输出：

* `checkpoints/best_model_v4_baseline_seed42_w5_dice003_area004.pt`
* `checkpoints/best_model_v4_calibrated_mu_seed42_w5_dice003_area004.pt`
* `results/metrics/v4_calibrated_mu_ablation.csv`
* `results/summaries/v4_calibrated_mu_ablation_summary.txt`
* `results/loss_curves/loss_curve_v4_calibrated_mu.png`
* `results/previews/reconstruction_preview_v4_calibrated_mu.png`

关键结论：

* `calibrated_mu` 相比 baseline seed=42 改善了 MSE、IoU、Dice、center_error、polygon area_error、small polygon IoU / Dice 和 multi_defect center_error；
* 缺陷区预测 μ_r 均值从约 399 降到约 361，中位数从约 295 降到约 262，说明输出校准方向有效但幅度有限；
* area_error 几乎不变，`pred_area > true_area` 数量反而增加；
* small polygon `pred_area=0` 仍为 0 / 25；
* 当前不切换全项目 baseline，推荐 baseline 仍以 `CURRENT_BASELINE.md` 为准。

下一步建议进入第 7.20B：在固定 seed 和同一 loss 配置下测试轻量 decoder 增强，判断 decoder 表达能力是否能进一步改善 μ 校准和面积误差。

---

## 最新实验状态：第 7.20B 步

第 7.20B 步已完成 `calibrated_mu` 轻量 decoder 增强实验。

本轮新增：

* `train_pinn.py` 支持 `--decoder-variant standard / enhanced`
* 默认 `standard`
* `standard` 保持旧 decoder：`128 / 128 / 64 + Tanh`
* `enhanced` 使用轻量增强 decoder：`256 / 256 / 128 / 64 + SiLU`

本轮没有修改 BzEncoder，没有修改 Fourier feature，没有加入新 loss，没有启用 physics_loss / L-BFGS，也没有修改标准评价指标定义。

主要输出：

* `checkpoints/best_model_v4_calibrated_mu_enhanced_decoder_seed42_w5_dice003_area004.pt`
* `results/metrics/v4_calibrated_mu_decoder_ablation.csv`
* `results/summaries/v4_calibrated_mu_decoder_ablation_summary.txt`
* `results/loss_curves/loss_curve_v4_calibrated_mu_enhanced_decoder.png`
* `results/previews/reconstruction_preview_v4_calibrated_mu_enhanced_decoder.png`

关键结论：

* enhanced decoder 让 defect_mu_mean 从约 361 降到约 333，defect_mu_median 从约 262 降到约 238，说明 decoder 容量会影响 μ 校准；
* enhanced decoder 小幅改善 Dice、small polygon IoU / Dice、small polygon IoU=0 数量和 multi_defect center_error；
* 但 area_error 从 0.6401 升到 0.9582，polygon area_error 从 0.7938 升到 1.4199，pred_area > true_area 从 182 / 200 增加到 189 / 200；
* 当前不切换全项目 baseline，推荐 baseline 仍以 `CURRENT_BASELINE.md` 为准。

下一步建议先做 seed repeat / 稳定性验证，确认 enhanced decoder 的校准收益和面积误差恶化是否稳定存在。
