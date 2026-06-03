# PINN_project 待删清单

这份文件最初用于审阅清理候选；2026-05-28 后已按用户确认执行多轮清理，历史记录见文末“已执行清理”章节。

清理边界：项目规则禁止批量删除。后续如果真的执行清理，只能删除用户确认过的明确路径，或者由用户手动删除。不要误删 `results/manifests/`、当前 true 3D RBC 数据包、当前推理 artifact，除非之后明确改变 baseline 路线。

## 当前复现链必须保留

这些文件在当前 true 3D RBC / liftoff-conditioned 缺陷预测链路上，必须保留：

- `COMSOL_DATA_REGISTRY.md`
- `CURRENT_BASELINE.md`
- `NEXT_STEP.md`
- `EXPERIMENT_LOG.md`
- `PINN优化路线.md`
- `术语说明.md`
- `REAL_DATA_INTAKE_SCHEMA.md`
- `scripts/` 中当前 true 3D RBC、liftoff、surface forward-refinement、surface multi-pit geometry-primary、internal-defect 代表性脚本
- `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json`
- `results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json`
- `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`
- `results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json`
- `results/manifests/real_data_internal_block_dry_run_manifest.json`
- `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz`
- `data/comsol_mfl/generated/true_3d_rbc_liftoff_aug_pack/true_3d_rbc_liftoff_aug_pack.npz`
- `checkpoints/true_3d_rbc_baseline_artifacts/`
- `checkpoints/true_3d_rbc_liftoff_adapter_artifacts/`

建议保留的当前代表性结果：

- `results/metrics/true_3d_rbc_formal_benchmark_20_77_seed_summary.csv`
- `results/metrics/true_3d_rbc_formal_benchmark_20_77_metrics.csv`
- `results/metrics/true_3d_rbc_formal_benchmark_comparison_matrix.csv`
- `results/metrics/true_3d_rbc_formal_benchmark_decision_matrix.csv`
- `results/metrics/true_3d_rbc_formal_liftoff_benchmark_metrics.csv`
- `results/metrics/true_3d_rbc_liftoff_conditioned_inference_smoke_metrics.csv`
- `results/metrics/true_3d_rbc_liftoff_conditioned_inference_by_liftoff.csv`
- `results/metrics/true_3d_rbc_real_data_manifest_dry_run_validation.csv`
- `results/summaries/true_3d_rbc_formal_benchmark_20_77_summary.txt`
- `results/summaries/true_3d_rbc_formal_liftoff_benchmark_summary.txt`
- `results/summaries/true_3d_rbc_liftoff_conditioned_inference_smoke_summary.txt`
- `results/summaries/true_3d_rbc_real_data_manifest_dry_run_summary.txt`

## 低风险本地清理候选

这些主要是本地缓存、图片或历史日志，不参与当前预测复现：

| 路径 | 约占空间 | 判断 |
|---|---:|---|
| `.tmp/` | 2.00 MB | 临时文件。 |
| `__pycache__/` | 0.09 MB | Python 字节码缓存。 |
| `results/previews/` | 117.73 MB | 不建议整目录删除；先保留少量代表 PNG，其余历史 gallery 可删。 |
| `results/loss_curves/` | 7.92 MB | 不建议整目录删除；先保留少量代表训练曲线，其余历史曲线可删。 |
| `results/archive/` | 1.27 MB | 旧日志和归档输出。 |

## 明确不清理

这些目录虽然不是模型运行的硬依赖，但属于用户工作状态、笔记或工具上下文，不纳入清理候选：

- `.claude/`
- `.claudian/`
- `.obsidian/`
- `notes/`

## PNG 和 loss curve 代表性保留建议

`results/previews/` 和 `results/loss_curves/` 适合“留样本、删大头”，不适合一刀切。

`results/previews/` 建议保留：

- `results/previews/true_3d_rbc_profile_primary_loss_gallery/`：保留当前 true 3D profile 代表图，建议留 8-12 张，覆盖 best profile、worst profile、high Dice / high profile error、curvature risk。
- `results/previews/mask_boundary_current_baseline/`：保留旧 2D boundary baseline 的代表图，建议留 3-5 张。
- `results/previews/mask_boundary_grid_candidate/` 或 `results/previews/mask_boundary_forward_consistency_lambda010/`：保留一个旧 2D comparator gallery，建议留 3-5 张。
- `results/previews/v4_smallpoly_area_error_worst10/` 或 `results/previews/v4_postprocess_examples/`：保留 v4 small polygon / postprocess 代表图，建议留 3-5 张。
- 如需展示 COMSOL data-domain 历史，可再保留 `results/previews/comsol_pilot_v9_baseline/` 或 `results/previews/comsol_three_component_multi_defect_pilot_v4_gate/` 中少量图。

`results/previews/` 其余目录可以作为删除候选，尤其是已停止路线的 gallery，例如 `shape_prior_latent_candidate/`、`mask_boundary_grid_refine_candidate/`、`mask_boundary_unet_decoder_candidate/`、`comsol_rect_rot_*` 等。

`results/loss_curves/` 建议保留：

- `loss_curve_tv_5e-6.png`：早期 simple TV baseline 代表。
- `loss_curve_v3_complex_tv_sweep_2e-6.png`：旧 v3_complex MSE-oriented reference 代表。
- `loss_curve_v3_complex_shape_aware_seed42_tv2e-6.png`：旧 v3_complex shape-aware 代表。
- `loss_curve_v3_complex_threshold_margin_seed42.png`：旧阈值/边界方向代表。
- `loss_curve_v4_smallpoly_w5_dice_0p03.png`：v4 small polygon 代表。
- `loss_curve_v4_w5_dice003_area004.png`：v4 area-aware 代表。
- `loss_curve_v4_calibrated_mu.png`：calibrated_mu 代表。

`results/loss_curves/` 其余同类 sweep 图、重复 seed 图、L-BFGS 小扫图可以删除。

## Checkpoint 清理候选

建议只保留：

- `checkpoints/true_3d_rbc_baseline_artifacts/`
- `checkpoints/true_3d_rbc_liftoff_adapter_artifacts/`

优先删除候选：

| 路径 | 约占空间 | 判断 |
|---|---:|---|
| `checkpoints/shape_prior_latent_candidate/` | 155.25 MB | 旧的已停止候选路线。 |
| `checkpoints/mask_boundary_grid_signal_features_candidate/` | 53.37 MB | 旧 2D mask / boundary 候选。 |
| `checkpoints/mask_boundary_grid_refine_candidate/` | 40.17 MB | 旧 2D mask / boundary 候选。 |
| `checkpoints/mask_boundary_forward_consistency_candidate/` | 40.00 MB | 旧 2D mask / boundary 候选。 |
| `checkpoints/mask_boundary_grid_candidate/` | 40.00 MB | 旧 2D mask / boundary 候选。 |
| `checkpoints/mask_boundary_grid_edge_candidate/` | 40.00 MB | 旧 2D mask / boundary 候选。 |
| `checkpoints/mask_to_bz_forward_surrogate/` | 33.10 MB | 旧 2D forward-consistency 支撑模型；不属于当前 true 3D baseline。 |
| `checkpoints/mask_boundary_forward_consistency_lambda_bracket/` | 26.67 MB | 旧调参实验。 |
| `checkpoints/mask_boundary_forward_consistency_lambda010/` | 26.67 MB | 旧 archived comparator。 |
| `checkpoints/mask_boundary_unet_decoder_candidate/` | 24.80 MB | 旧的已停止候选路线。 |
| `checkpoints/shape_type_conditional_boundary_candidate/` | 13.37 MB | 旧的已停止候选路线。 |

如果不准备重跑历史实验，也可以删除：

- `checkpoints/best_model*.pt`
- `checkpoints/best_model_v3_complex*.pt`
- `checkpoints/best_model_v4*.pt`
- `checkpoints/model_selection_3seed/`
- `checkpoints/model_selection_audit/`
- `checkpoints/macro_area_selection_audit/`
- `checkpoints/mask_boundary_candidate/`
- `checkpoints/mask_boundary_sdf_candidate/`
- `checkpoints/geometry_boundary_candidate/`
- `checkpoints/geometry_forward_consistency_candidate/`
- `checkpoints/starconvex_radial_shape_candidate/`
- `checkpoints/anisotropic_basis_forward_candidate/`
- `checkpoints/deformable_quad_forward_candidate/`
- `checkpoints/oracle_quad_supervised_candidate/`
- `checkpoints/profile_band_forward_candidate/`

## Data 清理候选

除非还要重跑旧实验，数据只建议保留当前 true 3D RBC 与 liftoff 两个数据包。

必须保留：

- `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz`
- `data/comsol_mfl/generated/true_3d_rbc_liftoff_aug_pack/true_3d_rbc_liftoff_aug_pack.npz`

可以作为历史代表保留：

- `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.npz`
- `data/training_data_v3_complex_train.npz`
- `data/training_data_v3_complex_val.npz`
- `data/training_data_v3_complex_test.npz`

删除候选：

- `data/training_data_train.npz`
- `data/training_data_val.npz`
- `data/training_data_test.npz`
- `data/training_data_v3_complex_multiliftoff_single_train.npz`
- `data/training_data_v3_complex_multiliftoff_single_val.npz`
- `data/training_data_v3_complex_multiliftoff_single_test.npz`
- `data/training_data_v3_complex_multiliftoff_train.npz`
- `data/training_data_v3_complex_multiliftoff_val.npz`
- `data/training_data_v3_complex_multiliftoff_test.npz`
- `data/training_data_v3_complex_overfit100_train.npz`
- `data/training_data_v3_complex_overfit100_val.npz`
- `data/training_data_v3_complex_overfit100_test.npz`
- `data/training_data_v3_complex_overfit30_train.npz`
- `data/training_data_v3_complex_overfit30_val.npz`
- `data/training_data_v3_complex_overfit30_test.npz`
- `data/training_data_v3_complex_small_os3_train.npz`
- `data/training_data_v4_balanced_complex_train.npz`
- `data/training_data_v4_balanced_complex_val.npz`
- `data/training_data_v4_balanced_complex_test.npz`
- `training_data_v2.npz`
- 所有不被当前 manifest 引用的旧 `data/comsol_mfl/prepared/comsol_*` 数据包
- 如果保留当前 assembled 数据包，可以删除旧 true 3D source / top-up 包：
  - `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v1.npz`
  - `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.npz`
  - `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v1_topup.npz`
  - `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74.npz`
  - `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76.npz`

## Results 清理候选

`results/metrics/` 和 `results/summaries/` 里很多文件已经被 Git 跟踪。删除这些文件属于仓库清理，不只是本地磁盘清理。

建议保留策略：

- 保留全部 `results/manifests/`。
- 保留全部 `results/templates/`。
- 保留上面列出的当前 true 3D RBC / liftoff / real-data 代表结果。
- 不再把 review 结果作为长期项目产物保存；`claude_review_*` 和 `review_*` 文档可以删除，后续 review 结论只保留在对话或需要人工沉淀的正式文档中。
- 旧 2D comparator 只保留少量代表记录：
  - `results/summaries/v3_complex_mask_boundary_forward_consistency_lambda010_summary.txt`
  - `results/summaries/v3_complex_mask_boundary_grid_candidate_summary.txt`
  - `results/summaries/v3_complex_mask_to_bz_forward_surrogate_summary.txt`
- v4 / small polygon 只保留少量代表记录：
  - `results/summaries/v4_smallpoly_loss_summary.txt`
  - `results/summaries/v4_smallpoly_weight_area_combo_summary.txt`
  - `results/summaries/v4_calibrated_mu_ablation_summary.txt`

在保留代表记录后，以下可以作为删除候选：

- 大部分 `results/metrics/comsol_*`
- 大部分 `results/metrics/v3_complex_*`
- 大部分 `results/metrics/v4_*`
- 体积较大的规划 CSV：
  - `results/metrics/true_3d_rbc_liftoff_aug_pack_plan.csv`
  - `results/metrics/true_3d_rbc_dataset_240_topup_plan.csv`
  - `results/metrics/true_3d_rbc_liftoff_sensor_offset_plan.csv`
  - `results/metrics/true_3d_rbc_dataset_120_topup_plan.csv`
  - `results/metrics/true_3d_rbc_pilot_pack_plan.csv`
- 如果 summary 足够，旧 profile / curvature 逐样本大表可以删除：
  - `results/metrics/true_3d_rbc_v3_240_curvature_candidate_profile_metrics.csv`
  - `results/metrics/true_3d_rbc_v3_240_curvature_refined_profile_metrics.csv`
  - `results/metrics/true_3d_rbc_v3_240_feature_fusion_candidate_profile_metrics.csv`
  - `results/metrics/true_3d_rbc_v3_240_neural_training_gate_profile_metrics.csv`
  - `results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_profile_metrics.csv`

## 建议清理顺序

1. 先清本地缓存和旧日志：`.tmp/`、`__pycache__/`、`results/archive/`。
2. 再精简图片和曲线：`results/previews/`、`results/loss_curves/` 先按上面的代表性保留建议留样，再删除其余历史产物。
3. 再清旧 checkpoint，只保留两个当前 true 3D RBC artifact 目录。
4. 再清旧 `data/` NPZ，只保留当前 v3_240、liftoff NPZ，以及少量代表性历史数据包。
5. 最后再决定是否清理 Git 已跟踪的 `results/metrics/` 和 `results/summaries/`。

最大空间回收来源：

- `checkpoints/`：总计约 842.52 MB；当前必须保留的 artifact 目录合计约 0.57 MB。
- `data/`：清理前约 516.78 MB；清理后保留当前 true 3D RBC NPZ 和少量代表性历史数据，合计约 24.14 MiB。
- `results/previews/`：约 117.73 MB，精选代表图后仍可回收大部分空间。
- `results/metrics/`：约 90.95 MB，但很多文件被 Git 跟踪。

## 2026-05-28 已执行清理

本次已按上面的保留策略清理 `results/previews/`、`results/loss_curves/`、`checkpoints/`、`data/` 和根目录旧 `training_data_v2.npz`。执行方式为逐个明确文件删除，未递归删除目录。

实际删除：17,304 个文件，约 1,454.40 MiB。

清理后保留：

- `results/previews/`：33 个文件，约 5.34 MiB。
- `results/loss_curves/`：7 个文件，约 0.66 MiB。
- `checkpoints/`：4 个文件，约 0.57 MiB。
- `data/`：6 个文件，约 24.14 MiB。

空目录暂未清理；它们基本不占空间，后续如需整理目录结构，应逐个明确目录处理。

## 2026-05-28 第二轮已执行清理

本轮清理目标是继续压缩历史实验记录，只保留当前 true 3D RBC 主线可复盘内容。

已删除：

- `results/archive/`：旧归档日志和旧评估 CSV，126 个文件，约 1.27 MiB。
- `.tmp/`：临时 smoke / selection 产物，21 个文件，约 2.00 MiB。
- `__pycache__/`：Python 字节码缓存，4 个文件，约 0.09 MiB。
- `results/metrics/`：旧 `comsol_*`、`v3_complex_*`、`v4_*`、通用旧 baseline 指标，以及 `true_3d_rbc*plan.csv` 大计划表；共 657 个文件，约 82.25 MiB。

本轮合计删除：808 个文件，约 85.61 MiB。

`results/metrics/` 清理后保留 199 个文件，约 8.70 MiB。保留边界为：

- `true_3d_rbc*` 且不是 `*plan.csv` 的当前主线指标、benchmark、artifact verification、gallery index、robustness / liftoff 小表。
- `comsol_true_3d_profile_capability_matrix.csv`。

清理后又移除空目录 1,846 个；`results/`、`checkpoints/`、`data/` 下已无空目录。

## 2026-05-28 第三轮已执行清理

本轮删除根目录早期 2D / synthetic 入口脚本：

- `train_pinn.py`
- `evaluate_pinn.py`
- `data_generator_v2.py`

删除原因：这三个脚本服务早期 simple / v3_complex / v4 2D mask 或 μ-field 路线；当前 true 3D RBC baseline 复现不依赖它们。保留影响：依赖这些旧入口的历史 2D 脚本不再保证可直接运行；当前 true 3D RBC 主线仍以 `scripts/` 下 manifest loader、artifact loader 和 liftoff-conditioned inference 脚本为入口。

## 2026-05-28 第四轮已执行清理

本轮清理 `results/summaries/`，边界是：删除 review 文档和大量旧实验 `.txt` 流水账，只保留当前 true 3D RBC / liftoff / real-data 主线、少量旧 2D comparator、v4 small polygon、COMSOL true 3D capability、Piao/RBC 对齐等代表性入口。

已删除：432 个 `.txt` 文件，约 1.12 MiB。其中包含 73 个 `claude_review_*` / `review_*` review 文档。

清理后 `results/summaries/` 保留 29 个文件，约 89.74 KiB：

- 24 个代表性 `.txt` summary。
- 5 个 `.md` 路线、合同或预处理说明。

后续默认不再保存 review 结果文档；如确有长期价值，应整理进 `CURRENT_BASELINE.md`、`EXPERIMENT_LOG.md` 或专门的 Markdown 说明，而不是继续堆在 `results/summaries/`。

## 2026-06-03 GitHub tracked 产物清理

本轮清理目标是压缩 GitHub `main` 中已经追踪的过程文件，不处理本机 ignored 的 `data/`、`checkpoints/`、`results/previews/` 或个人工具目录。

已删除：175 个 GitHub tracked 文件。

删除范围：

- `results/summaries/review_*.txt`
- `results/summaries/*preflight*`
- `results/summaries/*route_decision*`
- `results/summaries/*decision_summary*`
- `results/metrics/*epoch_log*`
- `results/metrics/*candidate_profile_metrics.csv`
- `results/metrics/*selected_predictions.csv`
- `results/metrics/*reference_predictions.csv`
- `results/metrics/*failure_cases.csv`
- `results/metrics/*gallery_index.csv`

保护范围：未删除 `CURRENT_BASELINE.md`、`COMSOL_DATA_REGISTRY.md`、`results/manifests/`、`results/templates/`，也未删除 25.18 / 25.19 / 25.19b completion package。清理后 GitHub tracked 文件数从 1221 降到 1046；`results/metrics/` 保留 435 个文件，`results/summaries/` 保留 174 个文件，`results/manifests/` 保留 43 个文件。

## 2026-06-03 第二轮 GitHub tracked 结构清理

本轮继续压缩 GitHub `main` 中与当前 25.19b 进度不符的历史入口。

已删除：96 个 GitHub tracked 文件。

删除范围：

- 根目录旧 COMSOL data-domain baseline 文档：
  - `COMSOL_DATA_BASELINE.md`
  - `COMSOL_DATA_BASELINE_V2.md`
  - `COMSOL_MULTI_DEFECT_DATA_BASELINE.md`
  - `COMSOL_THREE_COMPONENT_DATA_BASELINE.md`
- 92 个已停止路线脚本，覆盖旧 2D mask/boundary、v4/smallpoly、rect/rot profile-forward、multiheight/multiaxis/multidirection profile residual、starconvex、shape-prior、oracle/deformable、旧 COMSOL data-domain candidate 等实验入口。

保护范围：未删除 `CURRENT_BASELINE.md`、`COMSOL_DATA_REGISTRY.md`、`results/manifests/`、`results/templates/`、当前 true 3D RBC / liftoff / surface forward-refinement / surface multi-pit geometry-primary / internal-defect 代表性脚本，也未删除 25.18 / 25.19 / 25.19b completion package。清理后 GitHub tracked 文件数从 1046 降到 950，`scripts/` 从 372 个文件降到 280 个文件。

## 2026-06-03 第三轮 GitHub tracked metrics 清理

本轮继续压缩 GitHub `main` 中的非 internal 中间 metrics。用户明确要求保留“用 RBC 测 internal”的 internal-defect 部分，因此所有 `internal_defect*` / `internal_*` metrics 均保留。

已删除：103 个 GitHub tracked metrics 文件。

删除范围：

- NLS / surface RBC raw feature and correlation tables。
- 25.10-25.17 raster-target training / audit 大 JSON；25.18 route reset 和 25.19 / 25.19b handoff 保留。
- true 3D RBC v1 / v2_120 / v3_240 curvature / feature-fusion / Piao-NLS / profile-primary / pilot / topup / smoke 中间指标表。
- surface RBC targeted expansion 中间 plan / mesh / validation metrics。

保护范围：未删除任何 `internal_defect*` / `internal_*` metrics，未删除 25.18 / 25.19 / 25.19b completion package metrics，未删除 20.77 formal benchmark、formal liftoff benchmark、liftoff-conditioned inference、surface forward-refinement formal / inference 代表 metrics。清理后 GitHub tracked 文件数从 950 降到 847；`results/` 从 654 个文件降到 551 个文件；`results/metrics/` 从 435 个文件降到 332 个文件，约 4.76 MiB。
