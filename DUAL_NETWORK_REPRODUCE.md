# DUAL_NETWORK_REPRODUCE

## 1. 文档目的

本文档记录当前双网络变分支线关键实验的复现方式、推荐配置和结果入口。它不是新实验结果，也不改变现有训练代码；所有结论应结合 `DUAL_NETWORK_RESULTS_REPORT.md` 和已有 `experiments/dual_network/` 产物阅读。

## 2. 环境前提

- 需要 Python、numpy、torch。
- matplotlib 不是必须；当前 S31 图表使用 SVG，由 Python 标准库生成，不依赖 matplotlib。
- 推荐在项目根目录运行命令。
- 当前支线命令默认只处理 `feature/dual-network-variational`，不要把结果直接同步进 `main`。

## 3. 数据生成入口

数据由 `data_generator_v2.py` 生成。当前支线 runner 主要依赖 `.npz` 中的字段：

- `signals`
- `mu_maps`
- `x`
- `y`
- `metadata`

约定说明：

- `signals` 来自 `bz_signal[-1, :]`。
- 当前 probe line 使用 `y_s = 10.0`，与生成器的 `y_max` 对齐。
- `mu_maps` 主要用于诊断指标和半监督 / 诊断上界 prior，不应被表述为纯无监督 weak-form 训练目标。

## 4. Runner 入口

当前推荐使用：

- `train_dual_variational.py`

不建议继续用以下 prototype 作为多样本实验入口：

- `minimal_dual_single_sample_loop.py`

区别：

- `train_dual_variational.py` 用于多样本 runner，会对 `.npz` 中多个 sample 独立运行双网络 weak-form loop，并输出 `metrics.csv`。
- `minimal_dual_single_sample_loop.py` 保留为单样本 prototype，用于快速验证数据接口、交替优化和诊断输出。

## 5. 推荐配置

### 20x10 推荐配置

来源：

- `experiments/dual_network/S19_runner_50sample_bce_validation/`

配置：

- `grid_x=20`
- `grid_y=10`
- `samples=50`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `lambda_mask_bce_prior=1.0`
- `mask_prior_temperature=50.0`
- `center_mode=three`
- `test_radius=5.0`

结论：

`BCE mask prior` 显著优于 baseline，但它使用 `mu_label < 500` 的 mask，因此属于半监督 / 诊断上界。

### 40x20 推荐配置

来源：

- `experiments/dual_network/S24_40x20_50sample_default_validation/`

配置：

- `grid_x=40`
- `grid_y=20`
- `samples=50`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `lambda_mask_bce_prior=1.0`
- `mask_prior_temperature=25.0`
- `center_mode=three`
- `test_radius=5.0`

结论：

`temp25_lambda1` 是当前 40x20 的 IoU 优先默认候选。

### 80x40 推荐配置

来源：

- `experiments/dual_network/S28_80x40_50sample_default_validation/`

配置：

- `grid_x=80`
- `grid_y=40`
- `samples=50`
- `lambda_area_prior=1.0`
- `lambda_mask_prior=1.0`
- `lambda_mask_bce_prior=3.0`
- `mask_prior_temperature=25.0`
- `center_mode=three`
- `test_radius=5.0`

结论：

`temp25_lambda3` 是当前 80x40 综合默认候选；`temp20_lambda3` 可作为 IoU 优先参考。

## 6. 复现命令模板

以下命令是模板，不要求直接运行。请根据目标分辨率、样本数和输出目录替换占位参数。

### 6.1 生成数据命令模板

```powershell
python data_generator_v2.py `
  --train-samples <NUM_TRAIN> `
  --val-samples 0 `
  --test-samples 0 `
  --grid-x <GRID_X> `
  --grid-y <GRID_Y> `
  --output-dir experiments/dual_network/<RUN_NAME>/data `
  --seed <SEED>
```

### 6.2 baseline runner 命令模板

```powershell
python train_dual_variational.py `
  --npz-path experiments/dual_network/<RUN_NAME>/data/training_data_train.npz `
  --output-dir experiments/dual_network/<RUN_NAME>/baseline `
  --sample-indices 0,1,2 `
  --outer-steps 30 `
  --phi-steps 30 `
  --mu-steps 30 `
  --test-radius 5.0 `
  --center-mode three `
  --lambda-area-prior 1.0 `
  --lambda-mask-prior 1.0 `
  --lambda-mask-bce-prior 0.0 `
  --area-prior-temperature 50.0 `
  --mask-prior-temperature 50.0
```

### 6.3 BCE runner 命令模板

40x20 IoU 优先配置：

```powershell
python train_dual_variational.py `
  --npz-path experiments/dual_network/<RUN_NAME>/data/training_data_train.npz `
  --output-dir experiments/dual_network/<RUN_NAME>/temp25_lambda1 `
  --sample-indices 0,1,2 `
  --outer-steps 30 `
  --phi-steps 30 `
  --mu-steps 30 `
  --test-radius 5.0 `
  --center-mode three `
  --lambda-area-prior 1.0 `
  --lambda-mask-prior 1.0 `
  --lambda-mask-bce-prior 1.0 `
  --area-prior-temperature 50.0 `
  --mask-prior-temperature 25.0
```

80x40 综合候选配置：

```powershell
python train_dual_variational.py `
  --npz-path experiments/dual_network/<RUN_NAME>/data/training_data_train.npz `
  --output-dir experiments/dual_network/<RUN_NAME>/temp25_lambda3 `
  --sample-indices 0,1,2 `
  --outer-steps 30 `
  --phi-steps 30 `
  --mu-steps 30 `
  --test-radius 5.0 `
  --center-mode three `
  --lambda-area-prior 1.0 `
  --lambda-mask-prior 1.0 `
  --lambda-mask-bce-prior 3.0 `
  --area-prior-temperature 50.0 `
  --mask-prior-temperature 25.0
```

## 7. 结果查看入口

- `DUAL_NETWORK_RESULTS_REPORT.md`
- `DUAL_NETWORK_ARTIFACT_INDEX.md`
- `experiments/dual_network/S30_cross_resolution_report/aggregated_metrics.csv`
- `experiments/dual_network/S31_report_figures/`
- `experiments/dual_network/S29_80x40_visual_failure_report/summary.md`

## 8. 重要边界

- `BCE mask prior` 使用 `mu_label` mask。
- 当前结果是半监督 / 诊断上界。
- 当前不能声称纯无监督 weak-form 反演成功。
- 当前支线不替代 `main`。
- 不建议继续盲目扫描 `test_radius`、`center_mode` 或 `area prior`。

## 9. COMSOL parametric current candidate

This section records the current COMSOL parametric route candidate for the `feature/dual-network-variational` branch after S181-S185. It is a branch-local candidate, not a main baseline replacement.

Candidate:

- raw MLP signal encoder
- shared parametric head
- fixed-order component regression
- `center_representation=bin_offset`
- `center_bin_size_cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no validation-aware endpoint selection

Recommended reproduction command:

```powershell
python train_comsol_parametric_inverse.py `
  --train-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz `
  --train-targets experiments/dual_network/S113_comsol_parametric_targets/train/parametric_targets.npz `
  --val-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz `
  --val-targets experiments/dual_network/S113_comsol_parametric_targets/val/parametric_targets.npz `
  --test-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz `
  --test-targets experiments/dual_network/S113_comsol_parametric_targets/test/parametric_targets.npz `
  --output-dir experiments/dual_network/<RUN_NAME>/center_bin_offset_plus_grid_seed<N> `
  --steps 3000 `
  --lr 1e-3 `
  --hidden-dim 128 `
  --latent-dim 64 `
  --max-components 3 `
  --encoder-type mlp `
  --head-mode shared `
  --component-matching-mode fixed `
  --lambda-presence 1.0 `
  --lambda-type 1.0 `
  --lambda-continuous 1.0 `
  --center-representation bin_offset `
  --center-bin-size-cells 8 `
  --lambda-center-bin 1.0 `
  --lambda-center-offset 1.0 `
  --lambda-center-grid 0.1 `
  --lambda-center-axis-relative 0.0 `
  --lambda-raster-bce 0.0 `
  --lambda-raster-dice 0.0 `
  --val-selection-metric none `
  --val-selection-interval 0 `
  --seed <N> `
  --export-predictions
```

Do not use the forward-consistency runner for this candidate. Do not add raster loss, forward consistency, or validation-aware endpoint selection when reproducing the S185 candidate. Seed2/seed3 pass the S181-S185 stability gate but have lower val IoU than S179 seed1, so later center-bin stages should keep monitoring validation stability.

S186-S188 do not change the reproduction command. They consolidate this branch-local candidate and select `signal-to-center auxiliary head` as the next route for diagnosing remaining val center-bin instability.

S189-S193 tested the optional auxiliary head below as a diagnostic route. It did not outperform the same-round current-candidate reference and is not the current candidate:

```powershell
python train_comsol_parametric_inverse.py `
  --train-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz `
  --train-targets experiments/dual_network/S113_comsol_parametric_targets/train/parametric_targets.npz `
  --val-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz `
  --val-targets experiments/dual_network/S113_comsol_parametric_targets/val/parametric_targets.npz `
  --test-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz `
  --test-targets experiments/dual_network/S113_comsol_parametric_targets/test/parametric_targets.npz `
  --output-dir experiments/dual_network/<RUN_NAME>/aux_center_bin_offset_seed<N> `
  --steps 1500 `
  --lr 1e-3 `
  --hidden-dim 128 `
  --latent-dim 64 `
  --max-components 3 `
  --encoder-type mlp `
  --head-mode shared `
  --component-matching-mode fixed `
  --lambda-presence 1.0 `
  --lambda-type 1.0 `
  --lambda-continuous 1.0 `
  --center-representation bin_offset `
  --center-bin-size-cells 8 `
  --lambda-center-bin 1.0 `
  --lambda-center-offset 1.0 `
  --lambda-center-grid 0.1 `
  --lambda-center-axis-relative 0.0 `
  --aux-center-head `
  --lambda-aux-center-bin 1.0 `
  --lambda-aux-center-offset 1.0 `
  --aux-center-x-weight 1.0 `
  --aux-center-y-weight 1.0 `
  --lambda-raster-bce 0.0 `
  --lambda-raster-dice 0.0 `
  --val-selection-metric none `
  --val-selection-interval 0 `
  --seed <N> `
  --export-predictions
```

## 10. Historical COMSOL center-grid candidate

S170 previously promoted raw MLP / shared head / fixed-order + `lambda_center_grid=0.1` as the current branch candidate. S181-S185 supersedes it with `center_bin_offset_plus_grid`, but S170 remains the fallback reference if the center-bin route becomes unstable.

Historical S170 command:

```powershell
python train_comsol_parametric_inverse.py `
  --train-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz `
  --train-targets experiments/dual_network/S113_comsol_parametric_targets/train/parametric_targets.npz `
  --val-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz `
  --val-targets experiments/dual_network/S113_comsol_parametric_targets/val/parametric_targets.npz `
  --test-npz experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz `
  --test-targets experiments/dual_network/S113_comsol_parametric_targets/test/parametric_targets.npz `
  --output-dir experiments/dual_network/<RUN_NAME>/center_grid_candidate_seed<N> `
  --steps 3000 `
  --lr 1e-3 `
  --hidden-dim 128 `
  --latent-dim 64 `
  --max-components 3 `
  --encoder-type mlp `
  --head-mode shared `
  --component-matching-mode fixed `
  --lambda-presence 1.0 `
  --lambda-type 1.0 `
  --lambda-continuous 1.0 `
  --lambda-center-grid 0.1 `
  --lambda-center-axis-relative 0.0 `
  --lambda-raster-bce 0.0 `
  --lambda-raster-dice 0.0 `
  --val-selection-metric none `
  --val-selection-interval 0 `
  --seed <N> `
  --export-predictions
```
