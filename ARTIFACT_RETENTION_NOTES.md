# 代表性产物保留说明

这份说明记录本次清理后仍保留的非代码产物。目标是：保留当前缺陷预测复现链，同时为早期路线留少量可读代表，不再保留大量可重训但不打算重跑的历史 checkpoint / 数据包 / 图片。

## 当前主线必须保留

当前主线 = true 3D RBC profile/depth 缺陷预测。核心链路是：

`delta_b(Bx/By/Bz)` -> `20.77/20.85 Conv1D + MLP 六参数模型` -> `L_m/W_m/D_m/wLD/wWD/wLW` -> `3D profile/depth` -> `projected mask QA`

必须保留：

- `COMSOL_DATA_REGISTRY.md`：dataset_id 到 manifest / NPZ 的注册入口。
- `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json`：当前 v3_240 数据包 manifest。
- `results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json`：liftoff augmentation 数据包 manifest。
- `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`：20.77/20.85 baseline 推理 artifact 定位文件。
- `results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json`：A2 liftoff companion adapter 定位文件。
- `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz`：当前 nominal true 3D RBC 数据包。
- `data/comsol_mfl/generated/true_3d_rbc_liftoff_aug_pack/true_3d_rbc_liftoff_aug_pack.npz`：当前 liftoff robustness 数据包。
- `checkpoints/true_3d_rbc_baseline_artifacts/`：当前 baseline checkpoint 和 prediction artifact。
- `checkpoints/true_3d_rbc_liftoff_adapter_artifacts/`：当前 A2 adapter checkpoint 和 prediction artifact。

历史依据见：

- `CURRENT_BASELINE.md`：第 20.86 后的权威 baseline。
- `EXPERIMENT_LOG.md`：第 20.77、20.85、20.86、20.88a、20.96 等阶段记录。

## 保留的代表性 preview PNG

### 当前 true 3D profile/depth gallery

保留目录：

- `results/previews/true_3d_rbc_profile_primary_loss_gallery/`

保留理由：这是当前 true 3D profile/depth 路线最有解释力的图集，覆盖 best profile、worst profile、best Dice 但 profile 可能不佳、curvature risk，以及 3D surface true-vs-pred。这个目录本身已经是精选 gallery，因此整组保留。

### 旧 2D boundary baseline 代表

保留文件：

- `results/previews/mask_boundary_current_baseline/low_signal_improved_sample061_polygon.png`
- `results/previews/mask_boundary_current_baseline/mask_only_failure_sample141_polygon.png`
- `results/previews/mask_boundary_current_baseline/ordinary_medium_sample147_multi_defect.png`
- `results/previews/mask_boundary_current_baseline/ordinary_medium_sample193_rotated_rect.png`
- `results/previews/mask_boundary_current_baseline/small_polygon_improved_sample140_polygon.png`

保留理由：这些图代表旧 v3_complex 2D mask/boundary baseline 的典型收益和失败形态，足够支撑历史对比，不需要保留全量 gallery。

### 旧 2D grid candidate comparator 代表

保留文件：

- `results/previews/mask_boundary_grid_candidate/low_signal_improved_sample117_multi_defect.png`
- `results/previews/mask_boundary_grid_candidate/mask_boundary_grid_failure_sample037_polygon.png`
- `results/previews/mask_boundary_grid_candidate/ordinary_medium_sample017_rotated_rect.png`
- `results/previews/mask_boundary_grid_candidate/ordinary_medium_sample147_multi_defect.png`
- `results/previews/mask_boundary_grid_candidate/small_polygon_improved_sample054_polygon.png`

保留理由：这些图代表旧 grid decoder 候选的普通样本、低信号样本、多缺陷、polygon failure 和 small polygon 改善案例。

### v4 small polygon / postprocess 代表

保留文件：

- `results/previews/v4_postprocess_examples/postprocess_sample_033_thr300_min0.png`
- `results/previews/v4_postprocess_examples/postprocess_sample_113_thr300_min0.png`
- `results/previews/v4_postprocess_examples/postprocess_sample_122_thr300_min0.png`
- `results/previews/v4_postprocess_examples/postprocess_sample_130_thr300_min0.png`
- `results/previews/v4_postprocess_examples/postprocess_sample_184_thr300_min0.png`
- `results/previews/v4_smallpoly_area_error_worst10/area_error_worst_008_polygon.png`
- `results/previews/v4_smallpoly_area_error_worst10/area_error_worst_027_ellipse.png`
- `results/previews/v4_smallpoly_area_error_worst10/area_error_worst_130_polygon.png`
- `results/previews/v4_smallpoly_area_error_worst10/area_error_worst_142_polygon.png`
- `results/previews/v4_smallpoly_area_error_worst10/area_error_worst_172_polygon.png`

保留理由：v4 阶段主要证明 small polygon 漏检、面积高估和 threshold postprocess 的 trade-off。这 10 张图足够复盘机制，不再保留所有旧图。

## 保留的代表性 loss curve

保留文件：

- `results/loss_curves/loss_curve_tv_5e-6.png`
- `results/loss_curves/loss_curve_v3_complex_tv_sweep_2e-6.png`
- `results/loss_curves/loss_curve_v3_complex_shape_aware_seed42_tv2e-6.png`
- `results/loss_curves/loss_curve_v3_complex_threshold_margin_seed42.png`
- `results/loss_curves/loss_curve_v4_smallpoly_w5_dice_0p03.png`
- `results/loss_curves/loss_curve_v4_w5_dice003_area004.png`
- `results/loss_curves/loss_curve_v4_calibrated_mu.png`

保留理由：这 7 张覆盖 simple TV baseline、v3_complex MSE reference、v3 shape-aware / threshold 路线、v4 small polygon、v4 area-aware 和 calibrated_mu。其余同类 sweep、重复 seed、L-BFGS 小扫图不再保留。

## 保留的代表性历史数据

除当前 true 3D 主线数据外，保留少量早期数据包：

- `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.npz`
- `data/training_data_v3_complex_train.npz`
- `data/training_data_v3_complex_val.npz`
- `data/training_data_v3_complex_test.npz`

保留理由：v2_120 是 true 3D RBC v3_240 之前的代表性中间数据包；v3_complex train/val/test 是旧 2D boundary baseline 的代表数据集。其余旧数据包如需复现，应按脚本或外部 COMSOL 流程重新生成。

## 不保留旧 checkpoint 的原因

本次只保留当前推理链的两个 artifact 目录。旧 checkpoint 主要用于直接复跑历史模型；用户已明确不准备重跑历史实验，因此旧 checkpoint 删除后不影响当前预测复现。历史实验的机制、指标和结论仍由 `EXPERIMENT_LOG.md`、`CURRENT_BASELINE.md`、精简后的 `results/summaries/` 和少量代表 PNG / loss curve 保留。

## 保留的 summary 说明

`results/summaries/` 已不再作为 review 结果或历史流水账仓库。保留边界是：

- 当前 true 3D RBC / liftoff / real-data 的代表性 summary。
- 少量旧 2D comparator、v4 small polygon、COMSOL true 3D capability、Piao/RBC 对齐记录。
- 5 个 `.md` 路线、合同或预处理说明。

已删除 `claude_review_*`、`review_*`，以及大量 preflight、failure audit、training gate、route decision 的重复 `.txt` 记录。后续 review 结果默认不保存为项目产物；确有长期价值时，应整理进正式 Markdown 文档。

## 清理后实际规模

2026-05-28 清理后，本文档对应的非代码产物规模为：

- `results/previews/`：33 个文件。
- `results/loss_curves/`：7 个文件。
- `checkpoints/`：4 个文件，只剩当前 true 3D RBC baseline / liftoff adapter 两个 artifact 目录。
- `data/`：6 个文件，只剩当前 true 3D 主线数据和少量代表性历史数据。

第一轮清理后曾暂留空目录；第二轮已按用户确认删除空目录，保留说明以文件为准。

第二轮清理后继续保留：

- `results/metrics/`：199 个文件，约 8.70 MiB；只保留当前 true 3D RBC 主线非 plan 指标和 `comsol_true_3d_profile_capability_matrix.csv`。
- `results/summaries/`：29 个文件，约 89.74 KiB；只保留当前主线和少量历史代表入口，不再保留 review 文档。
- `results/manifests/`：11 个文件，约 0.03 MiB；保留为数据包和 artifact 定位入口。

`.tmp/`、`__pycache__/`、`results/archive/` 以及清理后空目录已删除。
