# DUAL_NETWORK_ARTIFACT_INDEX

## S48 conditional model artifacts

- `CONDITIONAL_DUAL_NETWORK_PLAN.md`: signal-conditioned dual-network 阶段设计文档，说明从 per-sample optimization 转向 `Bz signal + coords` 推理接口的动机、结构和边界。
- `conditional_dual_models.py`: conditional 模型骨架，包含 `BzEncoder`、`ConditionalMLP`、`ConditionalMuNet`、`ConditionalPhiNet` 和 `ConditionalDualNet`。
- `smoke_test_conditional_dual_models.py`: conditional 模型 smoke test，覆盖 `[N,2]` / `[B,N,2]` coords、输出 shape、`mu` 范围和错误 shape 的 `ValueError`。
- `experiments/dual_network/S48_conditional_model_skeleton/summary.md`: S48 阶段产物 summary。

## S49 conditional data interface artifacts

- `conditional_dual_data_utils.py`: conditional model batch data utilities，支持从 `.npz` 读取 `signals`、`mu_maps`、`coords` 或 `x/y`，并构造 `signals`、`coords`、`mu_label`、`mask_label` batch。
- `smoke_test_conditional_dual_data_utils.py`: tempfile-based smoke test，验证 conditional batch 构造和 `ConditionalDualNet` forward 集成。
- `experiments/dual_network/S49_conditional_data_interface/summary.md`: S49 阶段产物 summary。

## S50 conditional training runner artifacts

- `train_conditional_dual.py`: minimal conditional supervised training runner，使用 mask BCE / Dice / optional `mu_mse` 验证 `signals + coords -> mu / phi` 训练闭环。
- `smoke_test_train_conditional_dual.py`: tempfile-based smoke test，运行 5 step 小训练并检查 `metrics.csv` / `run_summary.md`。
- `experiments/dual_network/S50_conditional_training_runner_skeleton/summary.md`: S50 阶段产物 summary。

## S53 conditional train/val artifacts

- `experiments/dual_network/S53_conditional_train_val_probe/summary.md`: S53 train/val generalization probe summary，记录 20x10 train/val 数据、`big_bce_dice` / `big_bce_dice_mu1e-4` 的 train 与 val 指标，以及 train-val gap。

## S54 conditional train/val/test artifacts

- `experiments/dual_network/S54_conditional_train_val_test_probe/summary.md`: S54 train/val/test generalization probe summary, recording the 20x10 train / val / test split, `medium_bce_dice` / `big_bce_dice` metrics, and train-to-held-out gaps.

## S57 conditional signal normalization artifacts

- `experiments/dual_network/S57_conditional_signal_normalization_probe/summary.md`: S57 signal normalization probe summary, comparing `none`, `train_zscore`, and `per_sample_zscore` on 20x10 conditional train / val / test metrics.

## S58 conditional FiLM artifacts

- `experiments/dual_network/S58_conditional_film_probe/summary.md`: S58 conditioning architecture probe summary, comparing concat conditioning with FiLM conditioning under per-sample and train-global signal normalization.

## S59 conditional signal encoder artifacts

- `experiments/dual_network/S59_conditional_signal_encoder_probe/summary.md`: S59 signal encoder architecture probe summary, comparing MLP encoder, CNN encoder with concat conditioning, and CNN encoder with FiLM conditioning on 20x10 train / val / test metrics.

## S60 conditional local signal feature artifacts

- `experiments/dual_network/S60_conditional_local_signal_feature_probe/summary.md`: S60 coordinate-aligned local signal feature probe summary, comparing no local features, local Bz value, and local Bz value plus absolute value on 20x10 train / val / test metrics.

## S61 conditional direct mask head artifacts

- `experiments/dual_network/S61_conditional_direct_mask_head_probe/summary.md`: S61 direct mask head probe summary, comparing `mu_threshold` mask derivation with direct mask probability prediction on 20x10 train / val / test metrics.

## S62 conditional direct mask multi-task artifacts

- `experiments/dual_network/S62_conditional_direct_mask_multitask_probe/summary.md`: S62 direct mask multi-task probe summary, testing whether light `mu_mse` improves continuous `mu` errors for the direct mask head while preserving train / val / test mask IoU.

## S63 conditional derived Bz signal feature artifacts

- `experiments/dual_network/S63_conditional_derived_bz_feature_probe/summary.md`: S63 derived Bz signal feature probe summary, comparing raw single-Bz encoder input with `raw_abs_grad` derived features on 20x10 train / val / test metrics.

## S64 multi-height Bz interface artifacts

- `MULTI_HEIGHT_BZ_INTERFACE_PLAN.md`: design note for the multi-height / COMSOL-style Bz signal schema and first compatible flattening strategy.
- `experiments/dual_network/S64_multiheight_bz_interface/summary.md`: S64 interface skeleton summary, documenting 2D / 3D signals support and smoke test coverage.

## S65 synthetic multi-height proxy artifacts

- `build_multiheight_proxy_npz.py`: utility for converting single-channel Bz `.npz` files into synthetic three-channel smooth/decay proxy signals.
- `smoke_test_build_multiheight_proxy_npz.py`: tempfile smoke test for the proxy builder and conditional batch/model integration.
- `experiments/dual_network/S65_synthetic_multiheight_proxy_probe/summary.md`: S65 proxy probe summary, comparing S55 single-channel data against synthetic multi-channel proxy data.

## S66 COMSOL multi-height dataset interface artifacts

- `COMSOL_MULTIHEIGHT_BZ_DATA_PLAN.md`: schema and stage plan for real COMSOL / multi-height Bz `.npz` data.
- `comsol_multiheight_npz_utils.py`: validator and summary printer for COMSOL-style multi-height Bz `.npz` files.
- `smoke_test_comsol_multiheight_npz_utils.py`: mock COMSOL-style tempfile smoke test covering validator, conditional data utils, and model forward integration.
- `experiments/dual_network/S66_comsol_multiheight_dataset_interface/summary.md`: S66 interface summary and boundary notes.

## S67 COMSOL CSV to NPZ converter artifacts

- `convert_comsol_multiheight_csv_to_npz.py`：把 COMSOL-style long signal CSV 和 target `.npz` 转成兼容 S66 schema 的 multi-channel `.npz`。
- `smoke_test_convert_comsol_multiheight_csv_to_npz.py`：tempfile smoke test，覆盖 converter validation、converted NPZ schema validation、conditional batch flattening 和 model forward integration。
- `experiments/dual_network/S67_comsol_csv_to_npz_converter/summary.md`：S67 converter summary 和边界说明。

## 1. 顶层文档

- `README.md`：支线总览、运行方式和边界说明。
- `DUAL_NETWORK_TERMS.md`：术语说明，解释模型、loss、prior、指标和实验编号。
- `DUAL_NETWORK_EXPERIMENT_LOG.md`：实验日志，按 S3 之后的阶段记录实验目的、配置和结论。
- `DUAL_NETWORK_STAGE_SUMMARY.md`：阶段总结，整理当前能力、关键判断和下一步建议。
- `DUAL_NETWORK_RESULTS_REPORT.md`：跨分辨率结果报告，汇总 20x10、40x20、80x40 的核心指标。
- `DUAL_NETWORK_REPRODUCE.md`：复现实验命令与推荐配置说明。
- `DUAL_NETWORK_ARTIFACT_INDEX.md`：当前成果索引，用于快速定位文档、代码和关键实验产物。

## 2. 核心代码文件

- `dual_network_models.py`：定义 `PhiNet` 和 `MuNet`。
- `dual_network_losses.py`：定义 `energy_loss`、`data_loss`、`tv_loss`、`weak_form_loss` 和 compact-support `test_grads` 生成器。
- `dual_network_data_utils.py`：读取 `.npz` 数据，支持 `coords` 或 `x/y` 坐标来源，并构造 runner 输入。
- `train_dual_variational.py`：当前小规模 runner，对多个 sample 独立运行双网络 weak-form loop，并输出 `metrics.csv` 和诊断文件。
- `minimal_dual_single_sample_loop.py`：单样本 prototype，用于验证数据接口、交替优化和诊断指标。
- `evaluate_dual_variational.py`：支线评估脚本骨架，后续可扩展为正式可视化和评估入口。

说明：`train_dual_variational.py` 是当前主要实验入口，`minimal_dual_single_sample_loop.py` 是 prototype。当前不要把支线代码直接合并进 `main`。

## 3. 关键实验阶段

- S19：`experiments/dual_network/S19_runner_50sample_bce_validation/`
  - 用途：20x10 / 50-sample BCE validation。
  - 结论：BCE mask prior 在 20x10 下稳定优于 baseline，但属于半监督 / 诊断上界。

- S24：`experiments/dual_network/S24_40x20_50sample_default_validation/`
  - 用途：40x20 / 50-sample default validation。
  - 结论：`temp25_lambda1` 在 40x20 下显著优于 baseline，是当前 40x20 IoU 优先候选。

- S28：`experiments/dual_network/S28_80x40_50sample_default_validation/`
  - 用途：80x40 / 50-sample default validation。
  - 结论：`temp25_lambda3` 在 80x40 下稳定优于 baseline，是当前 80x40 综合候选。

- S29：`experiments/dual_network/S29_80x40_visual_failure_report/`
  - 用途：80x40 可视化失败诊断。
  - 结论：弱样本主要与形状细节、边界/窄缺陷、centroid 偏移和局部几何误差有关。

- S30：`experiments/dual_network/S30_cross_resolution_report/`
  - 用途：跨分辨率结果汇总。
  - 结论：BCE/default 配置在 20x10、40x20、80x40 均显著优于 baseline。

- S31：`experiments/dual_network/S31_report_figures/`
  - 用途：跨分辨率 SVG 图表。
  - 结论：图表化展示进一步支持 BCE/default 配置在各分辨率下优于 baseline。

## 4. 关键结果文件

- `experiments/dual_network/S30_cross_resolution_report/aggregated_metrics.csv`
- `experiments/dual_network/S30_cross_resolution_report/summary.md`
- `experiments/dual_network/S31_report_figures/defect_iou_by_resolution.svg`
- `experiments/dual_network/S31_report_figures/defect_area_pred_by_resolution.svg`
- `experiments/dual_network/S31_report_figures/mu_error_by_resolution.svg`
- `experiments/dual_network/S29_80x40_visual_failure_report/summary.md`

## 5. 当前阶段结论

- 双网络 weak-form 框架已经工程跑通。
- baseline 纯 weak-form / area / Dice 定位能力不足，容易产生低 `mu` 扩散和过大的 `defect_area_pred`。
- `BCE mask prior` 在 20x10、40x20、80x40 都明显优于 baseline。
- `BCE mask prior` 使用 `mu_label` mask，因此是半监督 / 诊断上界。
- 当前不能声称无监督 weak-form 反演成功。
- 支线当前最适合定位为“半监督双网络探索路线”。

## 6. 后续建议

下一步如果继续支线，应优先做结果整理、失败样本分类和论文式表述，而不是继续盲目扫描 `test_radius`、`center_mode` 或 `area prior`。
## S68 COMSOL pilot handoff artifacts

- `COMSOL_PILOT_DATA_REQUEST.md`：面向 COMSOL MCP / COMSOL 项目的真实 multi-height Bz pilot 数据请求说明。
- `experiments/dual_network/S68_comsol_pilot_handoff/summary.md`：S68 handoff summary，记录 pilot 建议规模、接入命令和边界。
## S69 COMSOL pilot handoff dry-run artifacts

- `smoke_test_comsol_pilot_handoff_end_to_end.py`：端到端 dry-run，模拟 COMSOL pilot export，覆盖 converter、validator、conditional batch 和 model forward。
- `experiments/dual_network/S69_comsol_pilot_handoff_dryrun/summary.md`：S69 dry-run summary，记录验证链路和边界。
## S70 COMSOL MCP prompt package artifacts

- `COMSOL_MCP_PILOT_PROMPT.md`：可直接交给 COMSOL MCP / COMSOL 相关对话的 pilot 数据生成任务提示。
- `experiments/dual_network/S70_comsol_mcp_prompt_package/summary.md`：S70 prompt package summary，记录使用方式和边界。

## S71 real COMSOL pilot ingest artifacts

- `experiments/dual_network/S71_comsol_pilot_ingest/raw/signals_multiheight.csv`：第一批真实 COMSOL pilot 的 multi-height Bz long CSV。
- `experiments/dual_network/S71_comsol_pilot_ingest/raw/targets.npz`：第一批真实 COMSOL pilot 的 `mu_maps` / `masks` target 文件。
- `experiments/dual_network/S71_comsol_pilot_ingest/raw/README.md`：COMSOL pilot 原始说明。
- `experiments/dual_network/S71_comsol_pilot_ingest/converted/comsol_multiheight_pilot.npz`：S67 converter 输出的支线可读 multi-channel NPZ。
- `experiments/dual_network/S71_comsol_pilot_ingest/summary.md`：S71 ingest summary，记录 converter、validator、conditional batch 和 model forward 检查。

## S73 COMSOL geometry-variation request artifacts

- `COMSOL_GEOMETRY_VARIATION_DATA_REQUEST.md`：下一批真实 COMSOL geometry-variation multi-height Bz 数据请求说明。
- `experiments/dual_network/S73_comsol_geometry_variation_request/summary.md`：S73 request summary，记录第一批 pilot 的边界、下一批建议规模和缺陷参数变化要求。


## S74 COMSOL geometry data ingest artifacts

- `experiments/dual_network/S74_comsol_geometry_data_ingest/summary.md`: S74 ingest summary，记录真实 COMSOL geometry train / val / test 数据接入、converter、validator、conditional batch 和 model forward 检查。
- `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/train_comsol_multiheight.npz`: S67 converter 输出的 train multi-channel NPZ。
- `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/val_comsol_multiheight.npz`: S67 converter 输出的 val multi-channel NPZ。
- `experiments/dual_network/S74_comsol_geometry_data_ingest/converted/test_comsol_multiheight.npz`: S67 converter 输出的 test multi-channel NPZ。
- `experiments/dual_network/S74_comsol_geometry_data_ingest/data_quality_summary.csv`: split 级数据质量摘要。
- `experiments/dual_network/S74_comsol_geometry_data_ingest/defect_param_summary.md`: 缺陷参数变化范围摘要。

## S75 COMSOL geometry conditional probe artifacts

- `experiments/dual_network/S75_comsol_geometry_conditional_probe/summary.md`: S75 第一轮真实 COMSOL multi-height conditional train / val / test probe summary。
- `experiments/dual_network/S75_comsol_geometry_conditional_probe/medium_multichannel/`: medium conditional runner 输出目录。
- `experiments/dual_network/S75_comsol_geometry_conditional_probe/big_multichannel/`: big conditional runner 输出目录。

## S76 COMSOL geometry probe summary artifacts

- `experiments/dual_network/S76_comsol_geometry_probe_summary/summary.md`: S74/S75 阶段总结、pilot 边界和下一步建议。


## S77 COMSOL target/mask diagnostics artifacts

- `comsol_target_mask_diagnostics.py`: 诊断 COMSOL `.npz` 中 `mu_maps` 与 `masks` 一致性的工具。
- `smoke_test_comsol_target_mask_diagnostics.py`: target/mask diagnostics 的 tempfile smoke test。
- `experiments/dual_network/S77_comsol_target_mask_diagnostics/summary.md`: S77 汇总，记录 train / val / test 的 mask consistency 结果。

## S78 COMSOL mask-source probe artifacts

- `experiments/dual_network/S78_comsol_mask_source_probe/summary.md`: S78 summary，比较 `mask_source=mu_threshold` 与 `mask_source=masks`。

## S79 COMSOL train-fit adaptation artifacts

- `experiments/dual_network/S79_comsol_train_fit_adaptation_probe/summary.md`: S79 summary，比较 `longer_steps`、`bigger_subsample` 和 `bce2_dice1`。

## S80 COMSOL target/mask and train-fit summary artifacts

- `experiments/dual_network/S80_comsol_target_and_fit_summary/summary.md`: S77-S79 阶段总结、瓶颈排序和下一步建议。


## S81 COMSOL direct mask / multitask output head artifacts

- `experiments/dual_network/S81_comsol_direct_mask_multitask_probe/summary.md`: S81 summary??? `mu_threshold_reference`?`direct_mu0` ? `direct_mu1e-5`?
- `experiments/dual_network/S81_comsol_direct_mask_multitask_probe/mu_threshold_reference/`: baseline runner ?????
- `experiments/dual_network/S81_comsol_direct_mask_multitask_probe/direct_mu0/`: direct mask head runner ?????
- `experiments/dual_network/S81_comsol_direct_mask_multitask_probe/direct_mu1e-5/`: direct mask head + light `mu_mse` runner ?????

## S82 COMSOL conditional bottleneck summary artifacts

- `experiments/dual_network/S82_comsol_conditional_bottleneck_summary/summary.md`: S74-S81 ?????????????

## S83 COMSOL geometry request V2 artifacts

- `COMSOL_GEOMETRY_VARIATION_DATA_REQUEST_V2.md`: ????? COMSOL geometry-variation multi-height Bz ???????
- `experiments/dual_network/S83_comsol_geometry_request_v2/summary.md`: S83 request summary?

## S84 COMSOL geometry V2 data ingest artifacts

- `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/summary.md`: S84 ingest summary，记录 V2 raw 文件复制、converter、validator、conditional batch 和 model forward 检查。
- `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/train_comsol_multiheight_v2.npz`: V2 train multi-channel NPZ。
- `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/val_comsol_multiheight_v2.npz`: V2 val multi-channel NPZ。
- `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/converted/test_comsol_multiheight_v2.npz`: V2 test multi-channel NPZ。
- `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/data_quality_summary.csv`: V2 split 级数据质量摘要。
- `experiments/dual_network/S84_comsol_geometry_v2_data_ingest/defect_param_summary.md`: V2 defect parameter 分布摘要。

## S85 COMSOL geometry V2 conditional probe artifacts

- `experiments/dual_network/S85_comsol_geometry_v2_conditional_probe/summary.md`: S85 V2 conditional train / val / test probe summary。
- `experiments/dual_network/S85_comsol_geometry_v2_conditional_probe/medium_multichannel_v2/`: medium V2 runner 输出目录。
- `experiments/dual_network/S85_comsol_geometry_v2_conditional_probe/big_multichannel_v2/`: big V2 runner 输出目录。

## S86 COMSOL geometry V2 probe summary artifacts

- `experiments/dual_network/S86_comsol_geometry_v2_probe_summary/summary.md`: S84/S85 阶段总结、V1 对比和下一步建议。

## S87 COMSOL V1/V2 target distribution diagnostics artifacts

- `comsol_v1_v2_target_distribution_diagnostics.py`: 比较 V1/V2 target、mask、label area 和 defect distribution 的诊断脚本。
- `smoke_test_comsol_v1_v2_target_distribution_diagnostics.py`: S87 tempfile smoke test。
- `experiments/dual_network/S87_comsol_v1_v2_target_distribution_diagnostics/summary.md`: S87 summary。
- `experiments/dual_network/S87_comsol_v1_v2_target_distribution_diagnostics/aggregate_target_distribution.csv`: V1/V2 split 级 target distribution 汇总。
- `experiments/dual_network/S87_comsol_v1_v2_target_distribution_diagnostics/defect_param_distribution.md`: V1/V2 defect parameter 分布摘要。

## S88 COMSOL signal semantics diagnostics artifacts

- `comsol_signal_semantics_diagnostics.py`: 比较 V1/V2 signal scale、lift-off 衰减和 offset 语义的诊断脚本。
- `smoke_test_comsol_signal_semantics_diagnostics.py`: S88 tempfile smoke test。
- `experiments/dual_network/S88_comsol_signal_semantics_diagnostics/summary.md`: S88 summary。
- `experiments/dual_network/S88_comsol_signal_semantics_diagnostics/aggregate_signal_semantics.csv`: V1/V2 split 级 signal 统计。
- `experiments/dual_network/S88_comsol_signal_semantics_diagnostics/lift_off_decay_diagnostics.csv`: lift-off 衰减诊断。

## S89 COMSOL signal interpretation probe artifacts

- `experiments/dual_network/S89_comsol_signal_interpretation_probe/S89_skipped.md`: S89 跳过说明。

## S90 COMSOL V2 data/signal diagnostic summary artifacts

- `experiments/dual_network/S90_comsol_v2_data_signal_diagnostic_summary/summary.md`: S87-S89 汇总结论和下一步建议。
## S93 COMSOL V2 small-label adaptation artifacts

- `experiments/dual_network/S93_comsol_v2_small_label_adaptation_probe/summary.md`: S93 summary，记录 `balanced_bce`、`balanced_pos_weight5` 和 `balanced_focal` 三组 V2 small-label adaptation 结果。
- `experiments/dual_network/S93_comsol_v2_small_label_adaptation_probe/balanced_bce/`: positive-balanced sampling + default BCE runner 输出。
- `experiments/dual_network/S93_comsol_v2_small_label_adaptation_probe/balanced_pos_weight5/`: positive-balanced sampling + `pos_weighted_bce` runner 输出。
- `experiments/dual_network/S93_comsol_v2_small_label_adaptation_probe/balanced_focal/`: positive-balanced sampling + `focal_bce` runner 输出。

## S94 COMSOL V2 small-label summary artifacts

- `experiments/dual_network/S94_comsol_v2_small_label_summary/summary.md`: S91-S93 阶段总结、瓶颈更新和下一步建议。
## S95 COMSOL V2 small-label failure summary artifacts

- `experiments/dual_network/S95_comsol_v2_small_label_failure_summary/summary.md`: S95 summary，记录 S93 全背景塌缩和当前最佳仍为 S85 big baseline。

## S97 COMSOL V1-to-V2 curriculum artifacts

- `experiments/dual_network/S97_comsol_v1_to_v2_curriculum_probe/summary.md`: S97 summary，比较 V2-only reproduce 与 V1 pretrain -> V2 finetune。
- `experiments/dual_network/S97_comsol_v1_to_v2_curriculum_probe/v2_only_baseline_reproduce/training_history.csv`: V2-only 训练动态记录。
- `experiments/dual_network/S97_comsol_v1_to_v2_curriculum_probe/v1_pretrain_v2_finetune/training_history.csv`: V1 pretrain / V2 finetune 训练动态记录。

## S98 COMSOL curriculum summary artifacts

- `experiments/dual_network/S98_comsol_curriculum_summary/summary.md`: S95-S97 阶段总结、curriculum 结论和下一步策略。

## S99 COMSOL V2 background-collapse summary artifacts

- `experiments/dual_network/S99_comsol_v2_background_collapse_summary/summary.md`: S99 summary，记录 S93 / S97 的全背景塌缩失败模式和下一步 area calibration 方向。

## S101 COMSOL V2 collapse suppression artifacts

- `experiments/dual_network/S101_comsol_v2_collapse_suppression_probe/summary.md`: S101 summary，比较 `v2_baseline_with_history`、`area_ratio_mse` 和 `foreground_floor`。
- `experiments/dual_network/S101_comsol_v2_collapse_suppression_probe/v2_baseline_with_history/training_history.csv`: baseline training history。
- `experiments/dual_network/S101_comsol_v2_collapse_suppression_probe/area_ratio_mse/training_history.csv`: area ratio MSE training history。
- `experiments/dual_network/S101_comsol_v2_collapse_suppression_probe/foreground_floor/training_history.csv`: foreground floor training history。

## S102 COMSOL V2 collapse suppression summary artifacts

- `experiments/dual_network/S102_comsol_v2_collapse_suppression_summary/summary.md`: S99-S101 阶段总结、瓶颈更新和下一步建议。

## S103 COMSOL threshold-margin diagnostics artifacts

- `comsol_threshold_margin_diagnostics.py`: 读取 runner 输出目录并诊断 hard area、soft foreground 和 `mu_threshold` crossing 的脚本。
- `smoke_test_comsol_threshold_margin_diagnostics.py`: S103 tempfile smoke test。
- `experiments/dual_network/S103_comsol_threshold_margin_diagnostics/summary.md`: S103 aggregate summary。
- `experiments/dual_network/S103_comsol_threshold_margin_diagnostics/*/threshold_margin_summary.csv`: 各 S101 run 的 threshold-margin 诊断摘要。

## S105 COMSOL V2 threshold-margin probe artifacts

- `experiments/dual_network/S105_comsol_v2_threshold_margin_probe/summary.md`: S105 summary，比较 baseline、positive-only margin 和 bidirectional margin。
- `experiments/dual_network/S105_comsol_v2_threshold_margin_probe/*/training_history.csv`: 各 S105 run 的 threshold-margin training history。

## S106 COMSOL V2 threshold-margin summary artifacts

- `experiments/dual_network/S106_comsol_v2_threshold_margin_summary/summary.md`: S103-S105 阶段总结、瓶颈更新和下一步建议。

## S107 COMSOL V2 threshold-margin stage summary artifacts

- `experiments/dual_network/S107_comsol_v2_threshold_margin_stage_summary/summary.md`: S107 summary，记录 hard threshold crossing 已恢复但 localization / shape 仍不足。

## S109 COMSOL V2 localization objective partial artifacts

- `experiments/dual_network/S109_comsol_v2_localization_objective_probe/summary.md`: S109 partial summary，记录已完成配置、提前停止原因和 `direct_mask_area_ratio` 未运行状态。
- `experiments/dual_network/S109_comsol_v2_localization_objective_probe/bidir_margin_val_select/`: validation-aware bidirectional margin run 输出。
- `experiments/dual_network/S109_comsol_v2_localization_objective_probe/bidir_margin_area_ratio/`: bidirectional margin + area ratio run 输出。
- `experiments/dual_network/S109_comsol_v2_localization_objective_probe/bidir_margin_floor/`: bidirectional margin + foreground floor run 输出。

## S110 COMSOL V2 localization objective stopped summary artifacts

- `experiments/dual_network/S110_comsol_v2_localization_objective_summary/summary.md`: S110 summary，明确 S109 long-run 搜索 partial / stopped 状态和 quick gate 切换原因。

## S111 COMSOL V2 quick diagnostic gate artifacts

- `COMSOL_V2_QUICK_DIAGNOSTIC_GATES.md`: COMSOL V2 后续 objective / output path 的 Gate 1 / Gate 2 / Gate 3 规则。
- `experiments/dual_network/S111_comsol_v2_quick_gate_protocol/summary.md`: S111 summary，记录 quick diagnostic gate protocol 和下一步 S112 建议。

## S112 COMSOL parametric inverse plan artifacts

- `COMSOL_PARAMETRIC_INVERSE_PLAN.md`: COMSOL V2 parametric inverse route 计划，说明 dense mask 转参数反演的动机、输出 schema、loss 和 rasterization 边界。
- `experiments/dual_network/S112_comsol_parametric_inverse_plan/summary.md`: S112 summary，记录路线切换原因和 S113-S116 计划。

## S113 COMSOL parametric target artifacts

- `comsol_parametric_targets.py`: 从 V2 `defect_params.csv` / NPZ metadata 构造 component-level parametric targets。
- `smoke_test_comsol_parametric_targets.py`: S113 tempfile smoke test。
- `experiments/dual_network/S113_comsol_parametric_targets/summary.md`: S113 aggregate summary。
- `experiments/dual_network/S113_comsol_parametric_targets/*/parametric_targets.npz`: train / val / test parametric targets。
- `experiments/dual_network/S113_comsol_parametric_targets/*/parametric_target_preview.csv`: target preview。
- `experiments/dual_network/S113_comsol_parametric_targets/*/parametric_target_summary.md`: split-level target summary。

## S114 COMSOL parametric inverse model artifacts

- `comsol_parametric_inverse_models.py`: Parametric Bz encoder、component head 和 `ParametricInverseNet`。
- `smoke_test_comsol_parametric_inverse_models.py`: S114 model smoke test。
- `experiments/dual_network/S114_comsol_parametric_inverse_model/summary.md`: S114 model skeleton summary。

## S115 COMSOL parametric inverse training artifacts

- `train_comsol_parametric_inverse.py`: V2 parametric inverse training runner。
- `smoke_test_train_comsol_parametric_inverse.py`: S115 runner smoke test。
- `experiments/dual_network/S115_comsol_parametric_inverse_training_probe/summary.md`: S115 training probe summary。
- `experiments/dual_network/S115_comsol_parametric_inverse_training_probe/v2_parametric_inverse/`: metrics、eval_metrics、test_metrics、training_history 和 run_summary 输出。

## S116 COMSOL parametric route summary artifacts

- `experiments/dual_network/S116_comsol_parametric_route_summary/summary.md`: S112-S115 parametric route 可行性总结和下一步建议。

## S117 COMSOL parametric raster oracle artifacts

- `comsol_parametric_rasterizer.py`: 从 component-level parametric targets rasterize 预测 mask，并计算 oracle IoU / Dice 的诊断脚本。
- `smoke_test_comsol_parametric_rasterizer.py`: S117 rotated rectangle、rotation 和 multi-component union smoke test。
- `experiments/dual_network/S117_comsol_parametric_raster_oracle/summary.md`: S117 train / val / test oracle rasterization gate 汇总。
- `experiments/dual_network/S117_comsol_parametric_raster_oracle/*/oracle_parametric_mask_metrics.csv`: split-level per-sample oracle mask metrics。
- `experiments/dual_network/S117_comsol_parametric_raster_oracle/*/oracle_parametric_mask_aggregate.csv`: split-level aggregate oracle mask metrics。

## S118 COMSOL refined parametric target artifacts

- `experiments/dual_network/S118_comsol_parametric_targets_refined/summary.md`: S118 angle sin/cos、continuous normalization 和 type distribution 汇总。
- `experiments/dual_network/S118_comsol_parametric_targets_refined/train/continuous_normalization_stats.npz`: train-only continuous normalization statistics。
- `experiments/dual_network/S118_comsol_parametric_targets_refined/*/parametric_targets.npz`: refined train / val / test parametric targets，包含 raw targets、normalized targets、target schema 和 angle encoding metadata。
- `experiments/dual_network/S118_comsol_parametric_targets_refined/*/parametric_target_preview.csv`: refined target preview。

## S119 COMSOL refined parametric inverse probe artifacts

- `experiments/dual_network/S119_comsol_parametric_inverse_refined_probe/summary.md`: S119 refined MLP probe summary，并记录 `refined_mlp_longer` 跳过原因。
- `experiments/dual_network/S119_comsol_parametric_inverse_refined_probe/refined_mlp/`: refined MLP metrics、eval_metrics、test_metrics、training_history 和 run_summary 输出。

## S120 COMSOL parametric route decision artifacts

- `experiments/dual_network/S120_comsol_parametric_route_decision/summary.md`: S117-S119 汇总结论、parametric route 是否继续和下一步建议。

## S121-S125 COMSOL parametric architecture diagnostics overview

- S121: parametric error decomposition，按 type、rotation、oracle gap 拆解 S115/S119 误差。
- S122: component-specific heads + stronger signal encoder，增加 `cnn1d`、`cnn1d_attention` 和 `component_specific` head。
- S123: quick gate architecture probe，比较四组 parametric architecture。
- S124: best architecture full probe，对 S123 内部最佳 `raw_cnn_component_specific` 执行 longer run。
- S125: route decision summary，判断当前最佳仍是 S115 raw parametric MLP baseline。

## S121 COMSOL parametric error diagnostics artifacts

- `comsol_parametric_error_diagnostics.py`: 读取 parametric run metrics 和 S117 oracle metrics，输出 oracle gap / type / rotation aggregate diagnostics。
- `smoke_test_comsol_parametric_error_diagnostics.py`: S121 tempfile smoke test。
- `experiments/dual_network/S121_comsol_parametric_error_diagnostics/summary.md`: S115 vs S119 error decomposition summary。
- `experiments/dual_network/S121_comsol_parametric_error_diagnostics/s115_raw/`: S115 raw diagnostics output。
- `experiments/dual_network/S121_comsol_parametric_error_diagnostics/s119_refined/`: S119 refined diagnostics output。

## S123 COMSOL parametric architecture quick gate artifacts

- `experiments/dual_network/S123_comsol_parametric_architecture_quick_gate/summary.md`: S123 four-config quick gate summary。
- `experiments/dual_network/S123_comsol_parametric_architecture_quick_gate/raw_mlp_shared_reference/`: MLP shared reference run output。
- `experiments/dual_network/S123_comsol_parametric_architecture_quick_gate/raw_mlp_component_specific/`: MLP component-specific run output。
- `experiments/dual_network/S123_comsol_parametric_architecture_quick_gate/raw_cnn_component_specific/`: CNN1D component-specific run output。
- `experiments/dual_network/S123_comsol_parametric_architecture_quick_gate/raw_attention_component_specific/`: CNN1D attention component-specific run output。

## S124 COMSOL parametric best architecture full probe artifacts

- `experiments/dual_network/S124_comsol_parametric_best_architecture_full_probe/summary.md`: S124 longer run summary。
- `experiments/dual_network/S124_comsol_parametric_best_architecture_full_probe/raw_cnn_component_specific_longer/`: selected longer run metrics and training history.

## S125 COMSOL parametric architecture decision artifacts

- `experiments/dual_network/S125_comsol_parametric_architecture_decision/summary.md`: S121-S124 route decision summary.

## S126 COMSOL parametric prediction export artifacts

- `train_comsol_parametric_inverse.py`: 新增 `--export-predictions` 和 `--component-matching-mode` 支持。
- `smoke_test_train_comsol_parametric_inverse.py`: 覆盖 prediction export 和 permutation matching smoke。
- `experiments/dual_network/S126_comsol_parametric_prediction_export/summary.md`: S126 prediction export summary。
- `experiments/dual_network/S126_comsol_parametric_prediction_export/s115_raw_mlp_export/`: S115 raw MLP export run，包含 metrics、training history、per-component prediction CSV 和 per-sample mask metrics。

## S127 COMSOL parametric grouped diagnostics artifacts

- `comsol_parametric_grouped_diagnostics.py`: 读取 S126 prediction export，按 type / slot / rotation / area / oracle gap 分组诊断。
- `smoke_test_comsol_parametric_grouped_diagnostics.py`: S127 tempfile smoke test。
- `experiments/dual_network/S127_comsol_parametric_grouped_diagnostics/summary.md`: S127 aggregate grouped diagnostics summary。
- `experiments/dual_network/S127_comsol_parametric_grouped_diagnostics/s115_raw_mlp_export/`: grouped CSV、worst samples 和 split/sample 诊断输出。

## S129 COMSOL parametric set-matching probe artifacts

- `experiments/dual_network/S129_comsol_parametric_set_matching_probe/summary.md`: fixed vs `permutation_min` probe summary。
- `experiments/dual_network/S129_comsol_parametric_set_matching_probe/fixed_reference/`: fixed-order reference metrics and prediction exports。
- `experiments/dual_network/S129_comsol_parametric_set_matching_probe/permutation_min/`: permutation-min matching metrics and prediction exports。

## S130 COMSOL parametric set-matching decision artifacts

- `experiments/dual_network/S130_comsol_parametric_set_matching_decision/summary.md`: S126-S129 route decision summary，记录 permutation matching 未改善，当前最佳仍是 S115 raw MLP baseline。

## S131 COMSOL parametric set-matching stage summary artifacts

- `experiments/dual_network/S131_comsol_parametric_set_matching_stage_summary/summary.md`: S126-S130 set-matching 阶段收束，确认下一步转向 differentiable raster mask supervision。

## S132 COMSOL differentiable rasterizer artifacts

- `comsol_differentiable_parametric_rasterizer.py`: PyTorch differentiable soft rasterizer 和 soft BCE / Dice helpers。
- `smoke_test_comsol_differentiable_parametric_rasterizer.py`: S132 rasterizer smoke test。
- `experiments/dual_network/S132_comsol_differentiable_rasterizer/summary.md`: S132 soft rasterizer 公式、axis / rotation 语义和边界说明。

## S133 COMSOL parametric raster loss support artifacts

- `experiments/dual_network/S133_comsol_parametric_raster_loss_support/summary.md`: S133 runner 接入 raster BCE / Dice loss 的说明。

## S134 COMSOL parametric raster-supervision probe artifacts

- `experiments/dual_network/S134_comsol_parametric_raster_supervision_probe/summary.md`: S134 four-config raster-supervision quick gate summary。
- `experiments/dual_network/S134_comsol_parametric_raster_supervision_probe/param_only_reference/`: parameter-only reference run output。
- `experiments/dual_network/S134_comsol_parametric_raster_supervision_probe/raster_dice1/`: Dice raster supervision run output。
- `experiments/dual_network/S134_comsol_parametric_raster_supervision_probe/raster_bce05_dice1/`: BCE + Dice raster supervision run output。
- `experiments/dual_network/S134_comsol_parametric_raster_supervision_probe/raster_dice1_soft2/`: Dice raster supervision with `softness_cells=2.0` run output。

## S135 COMSOL parametric raster-supervision decision artifacts

- `experiments/dual_network/S135_comsol_parametric_raster_supervision_decision/summary.md`: S131-S134 route decision summary，记录 raster supervision 有信号但未稳定超过 S115 baseline。

## S136 COMSOL parametric raster stage summary artifacts

- `experiments/dual_network/S136_comsol_parametric_raster_stage_summary/summary.md`: S131-S135 raster-supervision 阶段收束，确认下一步转向 two-stage raster fine-tune。

## S137 COMSOL two-stage raster support artifacts

- `train_comsol_parametric_inverse.py`: 新增 `--raster-loss-start-step`、`--val-selection-metric` 和 `--val-selection-interval`。
- `smoke_test_train_comsol_parametric_inverse.py`: 覆盖 raster schedule、validation selection、以及 delayed raster loss 与 `val_loss` selection 的非法组合。

## S138 COMSOL two-stage raster fine-tune probe artifacts

- `experiments/dual_network/S138_comsol_parametric_two_stage_raster_probe/summary.md`: S138 three-config two-stage raster fine-tune summary。
- `experiments/dual_network/S138_comsol_parametric_two_stage_raster_probe/param_only_val_select/`: parameter-only + validation selection reference run output。
- `experiments/dual_network/S138_comsol_parametric_two_stage_raster_probe/two_stage_raster_dice/`: delayed raster Dice fine-tune run output。
- `experiments/dual_network/S138_comsol_parametric_two_stage_raster_probe/two_stage_raster_bce_dice/`: delayed raster BCE + Dice fine-tune run output。

## S139 COMSOL two-stage raster decision artifacts

- `experiments/dual_network/S139_comsol_two_stage_raster_decision/summary.md`: S136-S138 route decision summary，记录 two-stage raster fine-tune 未稳定超过 S115 baseline。

## S140 COMSOL raster fine-tune stage summary artifacts

- `experiments/dual_network/S140_comsol_parametric_raster_finetune_stage_summary/summary.md`: S136-S139 raster fine-tune 阶段收束，确认下一步转向 physics-based signal features。

## S141 COMSOL MFL physics feature artifacts

- `comsol_mfl_physics_features.py`: 从 COMSOL multi-height Bz signals 中提取 peak、width、energy、lift-off decay ratio 和 channel correlation features。
- `smoke_test_comsol_mfl_physics_features.py`: S141 tempfile smoke test。
- `experiments/dual_network/S141_comsol_mfl_physics_features/summary.md`: S141 aggregate feature summary。
- `experiments/dual_network/S141_comsol_mfl_physics_features/*/physics_features.npz`: train / val / test physics feature arrays。
- `experiments/dual_network/S141_comsol_mfl_physics_features/*/physics_features.csv`: train / val / test feature tables。
- `experiments/dual_network/S141_comsol_mfl_physics_features/*/feature_summary.md`: split-level feature ranges。

## S142 COMSOL parametric feature fusion support artifacts

- `comsol_parametric_inverse_models.py`: 新增 `FeatureMLP` 和 `feature_fusion_mode=none|features_only|concat_latent`。
- `train_comsol_parametric_inverse.py`: 新增 feature NPZ loading、train-only feature normalization 和 runner fusion flags。
- `smoke_test_comsol_parametric_inverse_models.py`: 覆盖 feature fusion model shapes and errors。
- `smoke_test_train_comsol_parametric_inverse.py`: 覆盖 runner `concat_latent` feature fusion path。

## S143 COMSOL physics feature fusion probe artifacts

- `experiments/dual_network/S143_comsol_parametric_physics_feature_fusion_probe/summary.md`: S143 raw / features-only / raw+features quick gate summary。
- `experiments/dual_network/S143_comsol_parametric_physics_feature_fusion_probe/raw_signal_reference/`: raw signal reference metrics and prediction exports。
- `experiments/dual_network/S143_comsol_parametric_physics_feature_fusion_probe/physics_features_only/`: features-only metrics and prediction exports。
- `experiments/dual_network/S143_comsol_parametric_physics_feature_fusion_probe/raw_plus_physics_features/`: raw+features concat-latent metrics and prediction exports。

## S144 COMSOL physics feature route summary artifacts

- `experiments/dual_network/S144_comsol_physics_feature_route_summary/summary.md`: S140-S143 route decision summary，记录 direct physics feature fusion 未超过 raw MLP baseline。

## S145 COMSOL physics feature stage summary artifacts

- `experiments/dual_network/S145_comsol_parametric_physics_feature_stage_summary/summary.md`: S140-S144 stage summary，确认 direct physics feature concat 不作为默认，下一步转向 learned forward surrogate / forward consistency。

## S146 COMSOL parametric forward surrogate artifacts

- `comsol_parametric_forward_surrogate.py`: geometry vector builder、signal z-score helpers 和 MLP forward surrogate。
- `smoke_test_comsol_parametric_forward_surrogate.py`: S146 model smoke test，覆盖 geometry vector shape、forward shape 和 backward。
- `train_comsol_parametric_forward_surrogate.py`: learned forward surrogate training runner，输出 metrics / history / summary，不保存权重。
- `smoke_test_train_comsol_parametric_forward_surrogate.py`: S146 runner tempfile smoke test。
- `experiments/dual_network/S146_comsol_parametric_forward_surrogate/summary.md`: S146 implementation summary。

## S147 COMSOL parametric forward surrogate quality artifacts

- `experiments/dual_network/S147_comsol_parametric_forward_surrogate_quality/summary.md`: S147 forward surrogate quality gate summary。
- `experiments/dual_network/S147_comsol_parametric_forward_surrogate_quality/forward_surrogate_mlp/`: metrics / history / summary for the V2 forward surrogate quality run.

## S148 COMSOL parametric forward consistency support artifacts

- `train_comsol_parametric_inverse_forward_consistency.py`: in-memory forward surrogate pretrain + frozen forward-consistency inverse runner。
- `smoke_test_train_comsol_parametric_inverse_forward_consistency.py`: S148 tempfile smoke test。
- `experiments/dual_network/S148_comsol_parametric_forward_consistency_support/summary.md`: S148 support summary。

## S149 COMSOL parametric forward consistency probe artifacts

- `experiments/dual_network/S149_comsol_parametric_forward_consistency_probe/summary.md`: S149 param-only vs learned forward consistency comparison。
- `experiments/dual_network/S149_comsol_parametric_forward_consistency_probe/param_only_reference/`: parameter-only reference metrics。
- `experiments/dual_network/S149_comsol_parametric_forward_consistency_probe/forward_consistency_lambda01/`: `lambda_forward_consistency=0.1` metrics。
- `experiments/dual_network/S149_comsol_parametric_forward_consistency_probe/forward_consistency_lambda1/`: `lambda_forward_consistency=1.0` metrics。

## S150 COMSOL forward consistency route summary artifacts

- `experiments/dual_network/S150_comsol_forward_consistency_route_summary/summary.md`: S145-S149 route decision summary，记录 learned forward consistency 未超过 parameter-only baseline。

## S151 COMSOL forward consistency stage summary artifacts

- `experiments/dual_network/S151_comsol_forward_consistency_stage_summary/summary.md`: S145-S150 stage summary，确认 simple forward consistency loss 不作为默认，下一步转向 residual diagnostic 和 type/rotation targeted supervision。

## S152 COMSOL forward residual diagnostic artifacts

- `comsol_forward_residual_diagnostics.py`: learned forward residual sensitivity diagnostic runner。
- `smoke_test_comsol_forward_residual_diagnostics.py`: S152 tempfile smoke test。
- `experiments/dual_network/S152_comsol_forward_residual_diagnostics/summary.md`: val/test residual sensitivity aggregate summary。
- `experiments/dual_network/S152_comsol_forward_residual_diagnostics/val/`: val per-sample and aggregate residual CSVs。
- `experiments/dual_network/S152_comsol_forward_residual_diagnostics/test/`: test per-sample and aggregate residual CSVs。

## S154 COMSOL type/rotation targeted probe artifacts

- `experiments/dual_network/S154_comsol_type_rotation_targeted_probe/summary.md`: S154 type / rotation targeted loss comparison。
- `experiments/dual_network/S154_comsol_type_rotation_targeted_probe/param_only_reference/`: parameter-only reference metrics and prediction exports。
- `experiments/dual_network/S154_comsol_type_rotation_targeted_probe/type_extra/`: extra type CE metrics and prediction exports。
- `experiments/dual_network/S154_comsol_type_rotation_targeted_probe/rotation_extra/`: circular rotation extra loss metrics and prediction exports。
- `experiments/dual_network/S154_comsol_type_rotation_targeted_probe/type_rotation_extra/`: combined extra loss metrics and prediction exports。

## S155 COMSOL forward residual and type/rotation decision artifacts

- `experiments/dual_network/S155_comsol_forward_residual_type_rotation_decision/summary.md`: S151-S154 route decision summary。

## S156 COMSOL type/rotation loss stage summary artifacts

- `experiments/dual_network/S156_comsol_type_rotation_loss_stage_summary/summary.md`: S151-S155 stage summary，确认 simple type / rotation loss 没有稳定突破，下一步转向 oracle ablation。

## S157 COMSOL parametric oracle ablation artifacts

- `comsol_parametric_oracle_ablation.py`: 读取 S126 predictions 与 S113 targets，逐项替换 GT type / rotation / center / axis / depth / continuous 并复用 hard rasterizer 评估 mask IoU。
- `smoke_test_comsol_parametric_oracle_ablation.py`: S157 tempfile smoke test，覆盖 rotation oracle、`gt_all` sanity、type mismatch 重建和缺字段错误。
- `experiments/dual_network/S157_comsol_parametric_oracle_ablation/summary.md`: S157 implementation summary。

## S158 COMSOL parametric oracle ablation result artifacts

- `experiments/dual_network/S158_comsol_parametric_oracle_ablation_results/summary.md`: train / val / test oracle ablation aggregate summary。
- `experiments/dual_network/S158_comsol_parametric_oracle_ablation_results/train/`: train per-sample and aggregate oracle ablation CSVs。
- `experiments/dual_network/S158_comsol_parametric_oracle_ablation_results/val/`: val per-sample and aggregate oracle ablation CSVs。
- `experiments/dual_network/S158_comsol_parametric_oracle_ablation_results/test/`: test per-sample and aggregate oracle ablation CSVs。

## S159 COMSOL parametric oracle ablation decision artifacts

- `experiments/dual_network/S159_comsol_parametric_oracle_ablation_decision/summary.md`: S158 route decision summary，记录 center localization 是当前主要 final mask IoU bottleneck。

## S160 COMSOL center bottleneck summary artifacts

- `experiments/dual_network/S160_comsol_parametric_center_bottleneck_summary/summary.md`: S156-S159 summary，确认 center localization 是当前最大 parametric mask IoU bottleneck。

## S161 COMSOL center diagnostics artifacts

- `comsol_parametric_center_diagnostics.py`: 读取 S84 NPZ、S113 targets、S126 predictions 和 mask metrics，输出 center grid / axis-relative error 及 correlation diagnostics。
- `smoke_test_comsol_parametric_center_diagnostics.py`: S161 tempfile smoke test，覆盖 grid spacing、center error 和 summary outputs。
- `experiments/dual_network/S161_comsol_parametric_center_diagnostics/summary.md`: train / val / test center diagnostics aggregate summary。
- `experiments/dual_network/S161_comsol_parametric_center_diagnostics/train/`: train per-component, per-sample, grouped and correlation CSVs。
- `experiments/dual_network/S161_comsol_parametric_center_diagnostics/val/`: val per-component, per-sample, grouped and correlation CSVs。
- `experiments/dual_network/S161_comsol_parametric_center_diagnostics/test/`: test per-component, per-sample, grouped and correlation CSVs。

## S163 COMSOL center-aware quick gate artifacts

- `experiments/dual_network/S163_comsol_parametric_center_loss_probe/summary.md`: 1500-step param-only vs center-grid vs center-axis-relative quick gate summary。
- `experiments/dual_network/S163_comsol_parametric_center_loss_probe/param_only_1500_reference/`: same-run 1500-step parameter-only reference metrics and prediction exports。
- `experiments/dual_network/S163_comsol_parametric_center_loss_probe/center_grid_loss/`: `lambda_center_grid=0.1` metrics and prediction exports。
- `experiments/dual_network/S163_comsol_parametric_center_loss_probe/center_axis_relative/`: `lambda_center_axis_relative=1.0` metrics and prediction exports。

## S164 COMSOL center-aware full probe artifacts

- `experiments/dual_network/S164_comsol_parametric_center_loss_full_probe/summary.md`: 3000-step `center_grid_loss` confirmation summary。
- `experiments/dual_network/S164_comsol_parametric_center_loss_full_probe/center_grid_loss_3000/`: full probe metrics and prediction exports。

## S165 COMSOL center-localization decision artifacts

- `experiments/dual_network/S165_comsol_center_localization_decision/summary.md`: S160-S165 decision summary and next-step recommendation。
## S166 COMSOL center-grid stability stage summary artifacts

- `experiments/dual_network/S166_comsol_center_grid_stability_stage_summary/summary.md`: S160-S165 stage summary，说明 center-grid 需要稳定性验证后才能作为候选。

## S168 COMSOL center-grid stability repeat artifacts

- `experiments/dual_network/S168_comsol_center_grid_stability_repeat/summary.md`: S164 existing run + seed1/seed2 repeat summary。
- `experiments/dual_network/S168_comsol_center_grid_stability_repeat/center_grid_seed1/`: seed1 center-grid metrics and prediction exports。
- `experiments/dual_network/S168_comsol_center_grid_stability_repeat/center_grid_seed2/`: seed2 center-grid metrics and prediction exports。

## S169 COMSOL center-grid stability aggregate artifacts

- `experiments/dual_network/S169_comsol_center_grid_stability_summary/summary.md`: aggregate stability summary with per-run deltas and acceptance table。
- `experiments/dual_network/S169_comsol_center_grid_stability_summary/aggregate_stability_metrics.csv`: per-run val/test IoU, center_grid_mae and baseline deltas。
- `experiments/dual_network/S169_comsol_center_grid_stability_summary/acceptance_criteria.csv`: pass/fail acceptance criteria table。

## S170 COMSOL center-grid candidate decision artifacts

- `experiments/dual_network/S170_comsol_center_grid_candidate_decision/summary.md`: decision summary promoting `lambda_center_grid=0.1` as current COMSOL parametric route candidate。

## S171 COMSOL center-grid candidate consolidation artifacts

- `experiments/dual_network/S171_comsol_center_grid_candidate_consolidation/summary.md`: documentation-only consolidation of S166-S170 as the current branch COMSOL parametric candidate.

## S172 COMSOL parametric candidate reproduce command artifacts

- `experiments/dual_network/S172_comsol_parametric_candidate_reproduce_command/summary.md`: summary of the explicit candidate reproduction command.
- `DUAL_NETWORK_REPRODUCE.md`: includes the current COMSOL parametric candidate command.

## S173 COMSOL center-grid candidate docs update artifacts

- `experiments/dual_network/S173_comsol_center_grid_candidate_docs_update/summary.md`: summary of the documentation updates.

## S174 COMSOL center representation next route artifacts

- `experiments/dual_network/S174_comsol_center_representation_next_route/summary.md`: next-route decision selecting center-bin classification + offset.

## S175 COMSOL center-grid candidate stage decision artifacts

- `experiments/dual_network/S175_comsol_center_grid_candidate_stage_decision/summary.md`: final documentation-only decision summary.

## S176 COMSOL center-bin route summary artifacts

- `experiments/dual_network/S176_comsol_center_bin_route_summary/summary.md`: center-bin route preflight summary.

## S177 COMSOL center-bin offset support artifacts

- `experiments/dual_network/S177_comsol_center_bin_offset_support/summary.md`: implementation summary for optional center-bin + offset support.

## S178 COMSOL center-bin offset quick gate artifacts

- `experiments/dual_network/S178_comsol_center_bin_offset_quick_gate/summary.md`: 1500-step quick gate summary.
- `experiments/dual_network/S178_comsol_center_bin_offset_quick_gate/current_candidate_reference_1500_seed1/`: same-round current candidate reference metrics and prediction exports.
- `experiments/dual_network/S178_comsol_center_bin_offset_quick_gate/center_bin_offset/`: bin-offset-only metrics and prediction exports.
- `experiments/dual_network/S178_comsol_center_bin_offset_quick_gate/center_bin_offset_plus_grid/`: bin-offset plus decoded center-grid metrics and prediction exports.

## S179 COMSOL center-bin offset full confirm artifacts

- `experiments/dual_network/S179_comsol_center_bin_offset_full_confirm/summary.md`: 3000-step full confirm summary.
- `experiments/dual_network/S179_comsol_center_bin_offset_full_confirm/center_bin_offset_plus_grid_3000_seed1/`: full confirm metrics and prediction exports.

## S180 COMSOL center-bin offset decision artifacts

- `experiments/dual_network/S180_comsol_center_bin_offset_decision/summary.md`: route decision summary.

## S181 COMSOL center-bin offset plus grid stability stage summary artifacts

- `experiments/dual_network/S181_comsol_center_bin_offset_plus_grid_stability_stage_summary/summary.md`: stability-stage summary for validating `center_bin_offset_plus_grid`.

## S183 COMSOL center-bin offset plus grid stability repeat artifacts

- `experiments/dual_network/S183_comsol_center_bin_offset_plus_grid_stability/summary.md`: seed1/seed2/seed3 repeat summary.
- `experiments/dual_network/S183_comsol_center_bin_offset_plus_grid_stability/center_bin_offset_plus_grid_3000_seed2/`: seed2 metrics and prediction exports.
- `experiments/dual_network/S183_comsol_center_bin_offset_plus_grid_stability/center_bin_offset_plus_grid_3000_seed3/`: seed3 metrics and prediction exports.

## S184 COMSOL center-bin offset plus grid stability aggregate artifacts

- `experiments/dual_network/S184_comsol_center_bin_offset_plus_grid_stability_summary/summary.md`: aggregate stability summary comparing S170 historical range and S181-S185 runs.
- `experiments/dual_network/S184_comsol_center_bin_offset_plus_grid_stability_summary/aggregate_stability_metrics.csv`: S170 range plus per-run stability metrics.
- `experiments/dual_network/S184_comsol_center_bin_offset_plus_grid_stability_summary/acceptance_criteria.csv`: pass/fail acceptance table.

## S185 COMSOL center-bin offset plus grid candidate decision artifacts

- `experiments/dual_network/S185_comsol_center_bin_offset_plus_grid_candidate_decision/summary.md`: decision summary promoting `center_bin_offset_plus_grid` as the current branch COMSOL parametric candidate.

## S186 COMSOL center-bin candidate consolidation artifacts

- `experiments/dual_network/S186_comsol_center_bin_candidate_consolidation/summary.md`: consolidation summary for the current branch COMSOL parametric candidate.

## S187 COMSOL center-bin error diagnostics artifacts

- `experiments/dual_network/S187_comsol_center_bin_error_diagnostics/summary.md`: diagnostics summary from existing S179/S183/S184/S185 outputs.
- `experiments/dual_network/S187_comsol_center_bin_error_diagnostics/center_bin_error_summary.md`: same diagnostic narrative for direct lookup.
- `experiments/dual_network/S187_comsol_center_bin_error_diagnostics/center_bin_stability_table.csv`: run-level train/val/test stability table.
- `experiments/dual_network/S187_comsol_center_bin_error_diagnostics/per_component_center_bin_errors.csv`: reconstructed per-component x/y bin correctness and center error.
- `experiments/dual_network/S187_comsol_center_bin_error_diagnostics/sample_center_bin_error_summary.csv`: per-sample center-bin error summary joined with mask IoU.
- `experiments/dual_network/S187_comsol_center_bin_error_diagnostics/grouped_center_bin_errors.csv`: val/test slot-level center-bin aggregates.
- `experiments/dual_network/S187_comsol_center_bin_error_diagnostics/worst_val_samples.csv`: lowest-IoU val samples across center-bin seeds.

## S188 COMSOL center-bin next route decision artifacts

- `experiments/dual_network/S188_comsol_center_bin_next_route_decision/summary.md`: next-route decision selecting `signal-to-center auxiliary head`.

## S189 COMSOL signal-to-center auxiliary stage artifacts

- `experiments/dual_network/S189_comsol_signal_to_center_aux_stage_summary/summary.md`: stage summary for testing an optional signal-to-center auxiliary head.

## S190 COMSOL signal-to-center auxiliary support artifacts

- `comsol_parametric_inverse_models.py`: optional auxiliary center head support in `ParametricInverseNet`.
- `train_comsol_parametric_inverse.py`: optional auxiliary center loss and aux metrics/history/summary fields.
- `smoke_test_comsol_parametric_inverse_models.py`: model shape/default compatibility smoke coverage.
- `smoke_test_train_comsol_parametric_inverse.py`: runner smoke coverage for an auxiliary center run.
- `experiments/dual_network/S190_comsol_signal_to_center_aux_support/summary.md`: implementation support summary.

## S191 COMSOL signal-to-center auxiliary quick gate artifacts

- `experiments/dual_network/S191_comsol_signal_to_center_aux_quick_gate/summary.md`: same-round quick gate summary.
- `experiments/dual_network/S191_comsol_signal_to_center_aux_quick_gate/s191_quick_gate_metrics.csv`: train/val/test metrics table for the reference and auxiliary runs.
- `experiments/dual_network/S191_comsol_signal_to_center_aux_quick_gate/current_candidate_reference/`: 1500-step same-round current-candidate reference metrics and prediction exports.
- `experiments/dual_network/S191_comsol_signal_to_center_aux_quick_gate/aux_center_bin_offset/`: auxiliary bin/offset metrics and prediction exports.
- `experiments/dual_network/S191_comsol_signal_to_center_aux_quick_gate/aux_center_bin_offset_xweighted/`: x-weighted auxiliary bin/offset metrics and prediction exports.

## S192 COMSOL signal-to-center auxiliary full confirm artifacts

- `experiments/dual_network/S192_comsol_signal_to_center_aux_full_confirm/S192_skipped.md`: skipped full-confirm note because S191 did not pass gate.

## S193 COMSOL signal-to-center auxiliary decision artifacts

- `experiments/dual_network/S193_comsol_signal_to_center_aux_decision/summary.md`: decision summary keeping the S185 candidate and not promoting the auxiliary head.

## S194 COMSOL signal-to-center auxiliary failure summary artifacts

- `experiments/dual_network/S194_comsol_signal_to_center_aux_failure_summary/summary.md`: summary explaining why the signal-to-center auxiliary head does not continue.

## S195 COMSOL center-bin failure diagnostics artifacts

- `comsol_center_bin_failure_diagnostics.py`: read-only diagnostic script for existing center-bin prediction exports.
- `smoke_test_comsol_center_bin_failure_diagnostics.py`: smoke test for x-bin wrong, y-bin wrong, and both-correct diagnostic cases.
- `experiments/dual_network/S195_comsol_center_bin_failure_diagnostics/summary.md`: script implementation summary.

## S196 COMSOL center-bin failure diagnostics artifacts

- `experiments/dual_network/S196_comsol_center_bin_failure_diagnostics/summary.md`: aggregate diagnostic summary across the S191 reference and auxiliary runs.
- `experiments/dual_network/S196_comsol_center_bin_failure_diagnostics/aggregate_center_bin_failure_summary.csv`: compact run/split summary table.
- `experiments/dual_network/S196_comsol_center_bin_failure_diagnostics/current_candidate_reference/`: per-component, per-sample, grouped, and worst-sample diagnostics for the S191 reference.
- `experiments/dual_network/S196_comsol_center_bin_failure_diagnostics/aux_center_bin_offset/`: diagnostics for the plain auxiliary run.
- `experiments/dual_network/S196_comsol_center_bin_failure_diagnostics/aux_center_bin_offset_xweighted/`: diagnostics for the x-weighted auxiliary run.

## S197 COMSOL center-bin failure decision artifacts

- `experiments/dual_network/S197_comsol_center_bin_failure_decision/summary.md`: decision summary recommending center-x-bin focused calibration / hard-sample refinement.

## S198 COMSOL x-bin failure stage summary artifacts

- `experiments/dual_network/S198_comsol_x_bin_failure_stage_summary/summary.md`: stage summary for the x-bin calibration probe.

## S199 COMSOL main center-bin loss weighting artifacts

- `train_comsol_parametric_inverse.py`: optional main center-bin CE weights via `--center-bin-x-weight`, `--center-bin-y-weight`, and `--center-bin-slot-weights`.
- `smoke_test_train_comsol_parametric_inverse.py`: smoke coverage for weighted main center-bin loss and invalid slot-weight configuration.

## S200 COMSOL x-bin center calibration quick gate artifacts

- `experiments/dual_network/S200_comsol_x_bin_center_calibration_quick_gate/summary.md`: same-round quick gate summary.
- `experiments/dual_network/S200_comsol_x_bin_center_calibration_quick_gate/s200_quick_gate_metrics.csv`: compact metrics table for reference and weighted runs.
- `experiments/dual_network/S200_comsol_x_bin_center_calibration_quick_gate/current_candidate_reference/`: same-round S185 candidate reference metrics and prediction exports.
- `experiments/dual_network/S200_comsol_x_bin_center_calibration_quick_gate/x_bin_weighted/`: x-bin weighted run metrics and prediction exports.
- `experiments/dual_network/S200_comsol_x_bin_center_calibration_quick_gate/x_bin_slot_weighted/`: x-bin plus slot-weighted run metrics and prediction exports.
- `experiments/dual_network/S200_comsol_x_bin_center_calibration_quick_gate/diagnostics/`: center-bin failure diagnostics for the three S200 runs.

## S201 COMSOL x-bin center calibration full confirm artifacts

- `experiments/dual_network/S201_comsol_x_bin_center_calibration_full_confirm/S201_skipped.md`: skipped full-confirm note because S200 did not pass gate.
- `experiments/dual_network/S201_comsol_x_bin_center_calibration_full_confirm/summary.md`: skipped-stage summary.

## S202 COMSOL x-bin center calibration decision artifacts

- `experiments/dual_network/S202_comsol_x_bin_center_calibration_decision/summary.md`: decision summary keeping the S185 candidate and stopping simple x-bin / slot-weight sweeps.

## S203 COMSOL x-bin weighting failure summary artifacts

- `experiments/dual_network/S203_comsol_x_bin_weighting_failure_summary/summary.md`: summary freezing the S185 candidate and stopping simple x-bin / slot loss weighting.

## S204 COMSOL hard-sample center-bin package artifacts

- `experiments/dual_network/S204_comsol_hard_sample_center_bin_package/summary.md`: hard-sample package summary and failure taxonomy.
- `experiments/dual_network/S204_comsol_hard_sample_center_bin_package/hard_sample_summary.csv`: current-candidate held-out hard samples.
- `experiments/dual_network/S204_comsol_hard_sample_center_bin_package/hard_component_summary.csv`: component-level hard sample details across S200 runs.
- `experiments/dual_network/S204_comsol_hard_sample_center_bin_package/hard_sample_run_comparison.csv`: same hard-sample keys compared across S200 runs.
- `experiments/dual_network/S204_comsol_hard_sample_center_bin_package/run_delta_summary.csv`: sample-level deltas versus `current_candidate_reference`.

## S205 COMSOL V3 hard-case data request artifacts

- `COMSOL_V3_HARD_CASE_DATA_REQUEST.md`: V2-compatible hard-case COMSOL data request.
- `experiments/dual_network/S205_comsol_v3_hard_case_data_request/summary.md`: summary of the hard-case data request scope.

## S206 COMSOL hard-sample data-design decision artifacts

- `experiments/dual_network/S206_comsol_hard_sample_data_design_decision/summary.md`: decision summary keeping S185 frozen and recommending hard-case data design plus bins-correct low-IoU diagnostics.

## S207 COMSOL V3 hard-case pack preflight artifacts

- `COMSOL_V3_HARD_CASE_DATA_REQUEST.md`: concrete V2-compatible hard-case pilot request with default and fallback sample mixes.
- `experiments/dual_network/S207_comsol_v3_hard_case_pack_preflight/summary.md`: S207 preflight summary and ingest-before-training boundary.

## S208 COMSOL V3 hard-case ingest artifacts

- `experiments/dual_network/S208_comsol_v3_hard_case_ingest/raw/`: ingested copy of the real COMSOL V3 hard-case fallback export, excluding the source export root from version control.
- `experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted/train_comsol_v3_hard_case.npz`: converted train NPZ with signals shape `[30,3,200]`.
- `experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted/val_comsol_v3_hard_case.npz`: converted val NPZ with signals shape `[10,3,200]`.
- `experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted/test_comsol_v3_hard_case.npz`: converted test NPZ with signals shape `[10,3,200]`.
- `experiments/dual_network/S208_comsol_v3_hard_case_ingest/summary.md`: S208 ingest and conversion summary.

## S209 COMSOL V3 hard-case parametric target artifacts

- `experiments/dual_network/S209_comsol_v3_hard_case_parametric_targets/train/parametric_targets.npz`: train parametric targets.
- `experiments/dual_network/S209_comsol_v3_hard_case_parametric_targets/val/parametric_targets.npz`: val parametric targets.
- `experiments/dual_network/S209_comsol_v3_hard_case_parametric_targets/test/parametric_targets.npz`: test parametric targets.
- `experiments/dual_network/S209_comsol_v3_hard_case_parametric_targets/summary.md`: target schema and boundary summary.

## S210 COMSOL V3 hard-case oracle raster artifacts

- `experiments/dual_network/S210_comsol_v3_hard_case_oracle_raster/train/`: train oracle raster metrics.
- `experiments/dual_network/S210_comsol_v3_hard_case_oracle_raster/val/`: val oracle raster metrics.
- `experiments/dual_network/S210_comsol_v3_hard_case_oracle_raster/test/`: test oracle raster metrics.
- `experiments/dual_network/S210_comsol_v3_hard_case_oracle_raster/summary.md`: oracle gate summary.

## S211 COMSOL V3 hard-case ingest gate decision artifacts

- `experiments/dual_network/S211_comsol_v3_hard_case_ingest_gate_decision/summary.md`: decision summary allowing V3 hard-case candidate evaluation.

## S212 COMSOL V3 candidate evaluation setup artifacts

- `experiments/dual_network/S212_comsol_v3_candidate_eval_setup/summary.md`: setup summary for V3 hard-case candidate evaluation.

## S213 COMSOL V3 zero-shot evaluation artifacts

- `experiments/dual_network/S213_comsol_v3_candidate_zero_shot/summary.md`: skipped zero-shot summary documenting V2/V3 center-bin grid mismatch.

## S214 COMSOL V3 small-train probe artifacts

- `experiments/dual_network/S214_comsol_v3_small_train_probe/v3_train_candidate/`: V3 train candidate metrics and prediction exports.
- `experiments/dual_network/S214_comsol_v3_small_train_probe/v3_train_param_only_reference/`: V3 train param-only reference metrics and prediction exports.
- `experiments/dual_network/S214_comsol_v3_small_train_probe/summary.md`: V3 small-train quick-probe summary.

## S215 COMSOL V3 hard-case grouped diagnostic artifacts

- `comsol_v3_hard_case_grouped_diagnostics.py`: read-only grouping script for V3 prediction exports and hard-case labels.
- `smoke_test_comsol_v3_hard_case_grouped_diagnostics.py`: smoke test for grouped diagnostics.
- `experiments/dual_network/S215_comsol_v3_hard_case_grouped_diagnostics/grouped_by_hard_case_type.csv`: grouped V3 metrics by hard-case label.
- `experiments/dual_network/S215_comsol_v3_hard_case_grouped_diagnostics/worst_v3_samples.csv`: worst V3 samples by mask IoU and center error.
- `experiments/dual_network/S215_comsol_v3_hard_case_grouped_diagnostics/summary.md`: grouped diagnostic interpretation.

## S216 COMSOL V3 hard-case candidate evaluation decision artifacts

- `experiments/dual_network/S216_comsol_v3_hard_case_candidate_eval_decision/summary.md`: decision summary recommending V3 geometry coordinate harmonization before further candidate evaluation.

## S217 COMSOL V3 geometry convention audit artifacts

- `experiments/dual_network/S217_comsol_v3_geometry_convention_audit/audit_stats.json`: V2/V3 coordinate and bbox audit stats.
- `experiments/dual_network/S217_comsol_v3_geometry_convention_audit/summary.md`: coordinate convention audit summary.

## S218 COMSOL V3 geometry normalized artifacts

- `normalize_comsol_v3_geometry_convention.py`: normalization script mapping V3 raw x/y geometry to V2-compatible meter convention.
- `smoke_test_normalize_comsol_v3_geometry_convention.py`: smoke test for x/y center/axis transform and raw depth retention.
- `experiments/dual_network/S218_comsol_v3_geometry_normalized/converted/`: normalized V3 converted NPZ files.
- `experiments/dual_network/S218_comsol_v3_geometry_normalized/raw_normalized/`: normalized V3 defect parameter CSV files.
- `experiments/dual_network/S218_comsol_v3_geometry_normalized/summary.md`: normalization range and depth/z boundary summary.

## S219 COMSOL V3 normalized target and oracle artifacts

- `experiments/dual_network/S219_comsol_v3_normalized_parametric_targets/`: normalized train/val/test parametric targets.
- `experiments/dual_network/S219_comsol_v3_normalized_oracle_raster/`: normalized oracle raster metrics.
- `experiments/dual_network/S219_comsol_v3_normalized_parametric_targets/summary.md`: normalized target schema and oracle gate summary.

## S220 COMSOL V3 normalized runability artifacts

- `experiments/dual_network/S220_comsol_v3_normalized_zero_shot_runability/v2_train_to_v3_normalized_val_test/`: 5-step runability-gate metrics and prediction exports.
- `experiments/dual_network/S220_comsol_v3_normalized_zero_shot_runability/summary.md`: runability-only summary; performance interpretation deferred.

## S221 COMSOL V3 geometry normalization decision artifacts

- `experiments/dual_network/S221_comsol_v3_geometry_normalization_decision/summary.md`: decision summary allowing a future normalized-V3 candidate evaluation stage.

## S222 COMSOL normalized V3 candidate evaluation setup artifacts

- `experiments/dual_network/S222_comsol_v3_normalized_candidate_eval_setup/summary.md`: setup summary for normalized V3 hard-case performance evaluation.

## S223 COMSOL normalized V3 zero-shot artifacts

- `experiments/dual_network/S223_comsol_v3_normalized_zero_shot/v2_train_to_v3_normalized_val_test/`: V2-train to normalized-V3 val/test zero-shot metrics and prediction exports.
- `experiments/dual_network/S223_comsol_v3_normalized_zero_shot/summary.md`: zero-shot result summary.

## S224 COMSOL normalized V3 train quick probe artifacts

- `experiments/dual_network/S224_comsol_v3_normalized_train_probe/v3_normalized_train_candidate/`: normalized V3 train candidate metrics and prediction exports.
- `experiments/dual_network/S224_comsol_v3_normalized_train_probe/v3_normalized_train_param_only_reference/`: normalized V3 train continuous param-only reference metrics and prediction exports.
- `experiments/dual_network/S224_comsol_v3_normalized_train_probe/summary.md`: normalized V3 train quick-probe summary.

## S225 COMSOL normalized V3 hard-case grouped diagnostic artifacts

- `comsol_v3_hard_case_grouped_diagnostics.py`: grouped diagnostic script, now supporting split filtering with `--splits`.
- `experiments/dual_network/S225_comsol_v3_normalized_hard_case_grouped_diagnostics/grouped_by_hard_case_type.csv`: normalized V3 grouped metrics by hard-case label.
- `experiments/dual_network/S225_comsol_v3_normalized_hard_case_grouped_diagnostics/worst_v3_samples.csv`: worst normalized V3 samples.
- `experiments/dual_network/S225_comsol_v3_normalized_hard_case_grouped_diagnostics/summary.md`: grouped diagnostic interpretation.

## S226 COMSOL normalized V3 candidate evaluation decision artifacts

- `experiments/dual_network/S226_comsol_v3_normalized_candidate_eval_decision/summary.md`: decision summary recommending a larger real V3 hard-case pack with true rotated/multi-component coverage.

## S227 COMSOL normalized V3 signal-target sanity artifacts

- `comsol_v3_signal_target_sanity.py`: read-only sanity script for normalized V3 signal scale, target alignment, bbox alignment, and center-bin targets.
- `smoke_test_comsol_v3_signal_target_sanity.py`: smoke test for the sanity script.
- `experiments/dual_network/S227_comsol_v3_signal_target_sanity/per_sample_signal_target_sanity.csv`: per-sample sanity diagnostics.
- `experiments/dual_network/S227_comsol_v3_signal_target_sanity/split_signal_target_sanity.csv`: split-level sanity diagnostics.
- `experiments/dual_network/S227_comsol_v3_signal_target_sanity/summary.md`: signal-scale gate summary.

## S228-S230 COMSOL normalized V3 tiny-overfit skipped artifacts

- `make_comsol_parametric_subset_package.py`: helper for future tiny-overfit subset packages without modifying the training runner.
- `smoke_test_make_comsol_parametric_subset_package.py`: smoke test for subset package slicing and sample index renumbering.
- `experiments/dual_network/S228_comsol_v3_one_sample_tiny_overfit/summary.md`: one-sample tiny-overfit skipped summary.
- `experiments/dual_network/S229_comsol_v3_five_sample_tiny_overfit/summary.md`: five-sample tiny-overfit skipped summary.
- `experiments/dual_network/S230_comsol_v3_full_train_fit_gate/summary.md`: full-train fit gate skipped summary.

## S231 COMSOL normalized V3 tiny-overfit decision artifacts

- `experiments/dual_network/S231_comsol_v3_tiny_overfit_decision/summary.md`: decision summary pointing to COMSOL V3 signal export / scaling diagnosis before more training or more data.

## S232 COMSOL V3 repaired Bz signal smoke artifacts

- `ComsolV3BzSignalRepairMiniSmoke.java`: real COMSOL 3-sample smoke entry for the repaired near-defect anomaly / delta-Bz signal route.
- `experiments/dual_network/S232_comsol_v3_bz_signal_repair_3sample_smoke_summary/summary.md`: 3-sample smoke summary and next-step decision.
- `comsol_geometry_v3_bz_signal_repair_3sample_smoke/`: local smoke output directory with raw CSV/NPZ/stat artifacts; intentionally not staged by default because it is generated data, not a committed experiment-tree artifact.

## S233-S236 COMSOL repaired V3 hard-case ingest artifacts

- `experiments/dual_network/S233_comsol_v3_repaired_hard_case_ingest/raw/`: committed copy of the repaired V3 hard-case pack used for branch ingest.
- `experiments/dual_network/S233_comsol_v3_repaired_hard_case_ingest/converted/`: converted train/val/test repaired V3 NPZ files.
- `experiments/dual_network/S233_comsol_v3_repaired_hard_case_ingest/summary.md`: data source, shape, signal stats, and re-counted hard-case distribution summary.
- `experiments/dual_network/S234_comsol_v3_repaired_parametric_targets/`: repaired V3 parametric targets for train/val/test.
- `experiments/dual_network/S234_comsol_v3_repaired_parametric_targets/summary.md`: target schema and `type_vocab` summary.
- `experiments/dual_network/S235_comsol_v3_repaired_oracle_raster/`: oracle rasterization metrics for repaired V3 train/val/test.
- `experiments/dual_network/S235_comsol_v3_repaired_oracle_raster/summary.md`: oracle IoU gate summary.
- `experiments/dual_network/S236_comsol_v3_repaired_ingest_gate_decision/summary.md`: ingest-gate decision and next-stage recommendation.

## S237-S241 COMSOL repaired V3 candidate evaluation artifacts

- `experiments/dual_network/S237_comsol_v3_repaired_candidate_eval_setup/summary.md`: repaired V3 candidate evaluation setup and boundaries.
- `experiments/dual_network/S238_comsol_v3_repaired_zero_shot/summary.md`: V2-train to repaired-V3 zero-shot runability failure summary.
- `experiments/dual_network/S239_comsol_v3_repaired_train_probe/repaired_v3_train_candidate/`: repaired V3 train candidate metrics and prediction exports.
- `experiments/dual_network/S239_comsol_v3_repaired_train_probe/repaired_v3_train_param_only_reference/`: repaired V3 train param-only reference metrics and prediction exports.
- `experiments/dual_network/S239_comsol_v3_repaired_train_probe/summary.md`: repaired V3 train quick-probe summary.
- `experiments/dual_network/S240_comsol_v3_repaired_hard_case_grouped_diagnostics/`: grouped diagnostics by repaired V3 hard-case type.
- `experiments/dual_network/S241_comsol_v3_repaired_candidate_eval_decision/summary.md`: decision summary recommending a larger repaired V3 pack before candidate promotion or mixed training.

## S242-S246 COMSOL repaired V3 normalized evaluation artifacts

- `experiments/dual_network/S242_comsol_repaired_v3_normalized_ingest/converted/`: normalized repaired V3 train/val/test converted NPZ files.
- `experiments/dual_network/S242_comsol_repaired_v3_normalized_ingest/raw_normalized/`: normalized repaired V3 defect parameter CSV files.
- `experiments/dual_network/S242_comsol_repaired_v3_normalized_ingest/normalization_ranges.csv`: raw and normalized x/y, center, and axis ranges.
- `experiments/dual_network/S242_comsol_repaired_v3_normalized_ingest/summary.md`: repaired V3 coordinate-normalization summary.
- `experiments/dual_network/S243_comsol_repaired_v3_normalized_oracle_gate/parametric_targets/`: normalized repaired V3 parametric targets.
- `experiments/dual_network/S243_comsol_repaired_v3_normalized_oracle_gate/oracle_raster/`: normalized oracle raster metrics.
- `experiments/dual_network/S243_comsol_repaired_v3_normalized_oracle_gate/summary.md`: oracle gate summary.
- `experiments/dual_network/S244_comsol_repaired_v3_normalized_zero_shot/v2_train_to_repaired_v3_normalized/`: V2-train to normalized repaired V3 zero-shot metrics and prediction exports.
- `experiments/dual_network/S244_comsol_repaired_v3_normalized_zero_shot/summary.md`: zero-shot result summary.
- `experiments/dual_network/S245_comsol_repaired_v3_normalized_train_probe/`: normalized repaired V3 train quick-probe outputs.
- `experiments/dual_network/S245_comsol_repaired_v3_normalized_grouped_diagnostics/`: grouped diagnostics by normalized repaired V3 hard-case type.
- `experiments/dual_network/S246_comsol_repaired_v3_normalized_evaluation_decision/summary.md`: decision summary recommending a larger repaired V3 hard-case pack.

## S247-S253 COMSOL V3 polygon geometry route artifacts

- `COMSOL_V3_POLYGON_GEOMETRY_DATA_REQUEST.md`: COMSOL polygon/corner-point export schema request.
- `ComsolV3PolygonSmokeExport.java`: real COMSOL 3-sample polygon smoke entry; compiled/helper outputs are not staged.
- `build_comsol_v3_polygon_smoke_pack.py`: parser for the COMSOL polygon smoke stdout into a local smoke pack.
- `comsol_polygon_targets.py`: polygon target builder for `polygon_params.csv`.
- `comsol_polygon_rasterizer.py`: hard polygon oracle rasterizer.
- `smoke_test_comsol_polygon_rasterizer.py`: mock polygon target/rasterizer smoke test.
- `experiments/dual_network/S247_comsol_v3_polygon_route_summary/summary.md`: route summary and old-schema failure mechanism.
- `experiments/dual_network/S248_comsol_v3_polygon_export_schema_request/summary.md`: COMSOL polygon export contract summary.
- `experiments/dual_network/S249_comsol_v3_polygon_target_rasterizer/summary.md`: target/rasterizer implementation summary.
- `experiments/dual_network/S250_comsol_v3_polygon_mock_oracle_smoke/summary.md`: mock oracle smoke summary.
- `experiments/dual_network/S251_comsol_v3_polygon_true_geometry_smoke/summary.md`: true COMSOL 3-sample smoke summary.
- `experiments/dual_network/S252_comsol_v3_polygon_oracle_gate/summary.md`: polygon oracle gate summary.
- `experiments/dual_network/S252_comsol_v3_polygon_oracle_gate/oracle/polygon_oracle_metrics.csv`: small polygon oracle metrics table for the 3-sample smoke.
- `experiments/dual_network/S253_comsol_v3_polygon_geometry_route_decision/summary.md`: polygon route decision summary.
- `comsol_v3_polygon_geometry_3sample_smoke/`: local generated COMSOL smoke output; intentionally not staged as a committed raw export.

## S254-S258 COMSOL V3 polygon hard-case ingest artifacts

- `comsol_polygon_target_utils.py`: exports rasterizer-ready polygon targets from converted NPZ embedded polygon arrays plus wide `polygon_params.csv`.
- `smoke_test_comsol_polygon_target_utils.py`: smoke test for embedded polygon target export and hard rasterizer compatibility.
- `experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/raw/`: committed branch-local copy of the polygon-compatible repaired V3 pack.
- `experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/converted/`: converted train/val/test polygon V3 NPZ files.
- `experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/summary.md`: raw ingest summary.
- `experiments/dual_network/S255_comsol_v3_polygon_converted_validation/summary.md`: converted NPZ validation summary.
- `experiments/dual_network/S256_comsol_v3_polygon_targets/`: rasterizer-ready polygon targets for train/val/test.
- `experiments/dual_network/S256_comsol_v3_polygon_targets/summary.md`: polygon target schema and split coverage summary.
- `experiments/dual_network/S257_comsol_v3_polygon_oracle_raster/`: polygon oracle rasterization outputs for train/val/test.
- `experiments/dual_network/S257_comsol_v3_polygon_oracle_raster/summary.md`: oracle gate summary.
- `experiments/dual_network/S258_comsol_v3_polygon_ingest_gate_decision/summary.md`: ingest decision and next-stage recommendation.

## S259-S263 COMSOL V3 polygon inverse artifacts

- `comsol_polygon_inverse_models.py`: first supervised polygon inverse model with fixed `max_components=3` and `max_vertices=4`.
- `smoke_test_comsol_polygon_inverse_models.py`: shape, finite loss, and backward smoke test.
- `train_comsol_polygon_inverse.py`: independent polygon inverse runner for presence/type/vertex supervision.
- `smoke_test_train_comsol_polygon_inverse.py`: runner tempfile smoke test.
- `make_comsol_polygon_subset_package.py`: helper for tiny-overfit polygon subset packages.
- `smoke_test_make_comsol_polygon_subset_package.py`: subset helper smoke test.
- `experiments/dual_network/S259_comsol_v3_polygon_inverse_route_summary/summary.md`: route summary.
- `experiments/dual_network/S260_comsol_v3_polygon_inverse_model/summary.md`: model summary.
- `experiments/dual_network/S261_comsol_v3_polygon_inverse_runner/summary.md`: runner summary.
- `experiments/dual_network/S262_comsol_v3_polygon_tiny_overfit/`: one-sample overfit outputs and skipped larger-gate summary.
- `experiments/dual_network/S263_comsol_v3_polygon_inverse_decision/summary.md`: decision summary.

## S264-S268 COMSOL V3 polygon raster-sensitivity repair artifacts

- `comsol_polygon_raster_sensitivity_diagnostics.py`: read-only diagnostic for vertex error in grid-cell units, raster area drift, pixel disagreement, and target-vertex oracle checks.
- `smoke_test_comsol_polygon_raster_sensitivity_diagnostics.py`: smoke test for the raster-sensitivity diagnostic.
- `train_comsol_polygon_inverse.py`: now supports default-off `--vertex-loss-space`, `--lambda-area-aux`, and `--lambda-edge-aux` repair knobs.
- `smoke_test_train_comsol_polygon_inverse.py`: runner smoke covering grid vertex loss and area/edge auxiliary fields.
- `experiments/dual_network/S264_comsol_v3_polygon_one_sample_failure_summary/summary.md`: S262 failure summary.
- `experiments/dual_network/S265_comsol_v3_polygon_raster_sensitivity_diagnostic/`: S262 vertex-to-raster diagnostic outputs.
- `experiments/dual_network/S266_comsol_v3_polygon_loss_repair_support/summary.md`: runner repair support summary.
- `experiments/dual_network/S267_comsol_v3_polygon_one_sample_repair_probe/`: one-sample repair probe and longer-overfit sensitivity outputs.
- `experiments/dual_network/S268_comsol_v3_polygon_one_sample_repair_decision/summary.md`: decision summary allowing the next 5-sample polygon gate.

## S269-S273 COMSOL V3 polygon 5-sample overfit artifacts

- `experiments/dual_network/S269_comsol_v3_polygon_5sample_overfit_setup/summary.md`: setup summary and S267 configuration reuse.
- `experiments/dual_network/S270_comsol_v3_polygon_5sample_subset/train_5sample_polygon_v3.npz`: 5-sample converted subset.
- `experiments/dual_network/S270_comsol_v3_polygon_5sample_subset/train_5sample_polygon_targets.npz`: 5-sample polygon targets.
- `experiments/dual_network/S270_comsol_v3_polygon_5sample_subset/hard_case_coverage.csv`: source-index and hard-case coverage table.
- `experiments/dual_network/S270_comsol_v3_polygon_5sample_subset/summary.md`: subset summary.
- `experiments/dual_network/S271_comsol_v3_polygon_5sample_overfit/five_sample_overfit/`: 5-sample overfit metrics and prediction exports.
- `experiments/dual_network/S271_comsol_v3_polygon_5sample_overfit/summary.md`: 5-sample aggregate overfit summary.
- `experiments/dual_network/S272_comsol_v3_polygon_5sample_diagnostics/`: per-sample, grouped, worst-sample, and raster-sensitivity diagnostics.
- `experiments/dual_network/S273_comsol_v3_polygon_5sample_decision/summary.md`: decision summary clearing the 5-sample gate.

## S274-S278 COMSOL V3 polygon train30 quick probe artifacts

- `experiments/dual_network/S274_comsol_v3_polygon_train30_quick_probe_setup/summary.md`: train30 setup and boundary summary.
- `experiments/dual_network/S275_comsol_v3_polygon_train30_quick_probe/train30_polygon_inverse/`: train30/val10/test10 metrics and prediction exports.
- `experiments/dual_network/S275_comsol_v3_polygon_train30_quick_probe/summary.md`: aggregate train30 quick-probe summary.
- `experiments/dual_network/S276_comsol_v3_polygon_train30_grouped_diagnostics/`: hard-case grouped diagnostics, worst samples, and raster-sensitivity outputs.
- `experiments/dual_network/S277_comsol_v3_polygon_train30_quick_probe_decision/summary.md`: decision summary marking train30 gate failure.
- `experiments/dual_network/S278_comsol_v3_polygon_train30_docs/summary.md`: documentation sync summary.

## S279-S283 COMSOL V3 polygon train30 fit repair artifacts

- `experiments/dual_network/S279_comsol_v3_polygon_train30_failure_summary/summary.md`: train30 failure summary and boundary.
- `experiments/dual_network/S280_comsol_v3_polygon_train30_failure_diagnostics/`: diagnostic-only tables reusing S275/S276 outputs.
- `experiments/dual_network/S281_comsol_v3_polygon_train30_repair_support/summary.md`: confirmation that no runner/model code change is needed.
- `experiments/dual_network/S282_comsol_v3_polygon_train30_repair_quick_gate/longer_train30/`: passing longer train30 repair run with prediction exports.
- `experiments/dual_network/S282_comsol_v3_polygon_train30_repair_quick_gate/longer_train30_train_raster_sensitivity/`: train raster-sensitivity diagnostics for the passing run.
- `experiments/dual_network/S282_comsol_v3_polygon_train30_repair_quick_gate/summary.md`: repair matrix and stop-on-pass summary.
- `experiments/dual_network/S283_comsol_v3_polygon_train30_repair_decision/summary.md`: decision summary clearing the train30 fit gate.

## S284-S288 COMSOL V3 polygon generalization diagnostics artifacts

- `comsol_polygon_generalization_diagnostics.py`: read-only split distribution and polygon prediction failure diagnostic.
- `smoke_test_comsol_polygon_generalization_diagnostics.py`: smoke test for the diagnostic script.
- `experiments/dual_network/S284_comsol_v3_polygon_generalization_failure_summary/summary.md`: generalization failure setup summary.
- `experiments/dual_network/S285_comsol_v3_polygon_generalization_distribution_diagnostics/`: split-level geometry/signal diagnostics and joined diagnostic tables.
- `experiments/dual_network/S286_comsol_v3_polygon_prediction_failure_diagnostics/`: prediction failure tables, grouped failures, and worst val/test samples.
- `experiments/dual_network/S287_comsol_v3_polygon_generalization_bottleneck_summary/summary.md`: bottleneck synthesis.
- `experiments/dual_network/S288_comsol_v3_polygon_generalization_decision/summary.md`: decision summary and next-stage recommendation.

## S289-S293 COMSOL V3 center-anchored polygon artifacts

- `comsol_center_anchored_polygon_targets.py`: center-bin plus local-vertex target builder for polygon V3 targets.
- `smoke_test_comsol_center_anchored_polygon_targets.py`: target encode/decode and oracle smoke test.
- `comsol_center_anchored_polygon_inverse_models.py`: center-anchored polygon inverse model.
- `smoke_test_comsol_center_anchored_polygon_inverse_models.py`: model shape and backward smoke test.
- `train_comsol_center_anchored_polygon_inverse.py`: independent center-anchored polygon inverse runner.
- `smoke_test_train_comsol_center_anchored_polygon_inverse.py`: runner smoke test.
- `make_comsol_center_anchored_polygon_subset_package.py`: subset helper for staged overfit gates.
- `experiments/dual_network/S289_comsol_v3_center_anchored_polygon_route_summary/summary.md`: route summary.
- `experiments/dual_network/S290_comsol_v3_center_anchored_polygon_targets/`: train/val/test center-anchored targets and decode-oracle metrics.
- `experiments/dual_network/S291_comsol_v3_center_anchored_polygon_runner/summary.md`: runner summary.
- `experiments/dual_network/S292_comsol_v3_center_anchored_polygon_gates/`: one-sample, five-sample, and train30 gate outputs plus grouped diagnostics.
- `experiments/dual_network/S293_comsol_v3_center_anchored_polygon_decision/summary.md`: decision summary.

## S294-S297 COMSOL V3 center-anchored held-out diagnostic artifacts

- `center_anchored_polygon_failure_diagnostics.py`: read-only diagnostic joining S292 center-anchored prediction exports with S290 targets and S254 metadata.
- `smoke_test_center_anchored_polygon_failure_diagnostics.py`: tempfile smoke test for the center-bin, local-shape, and matched-coverage diagnostic outputs.
- `experiments/dual_network/S294_center_anchored_polygon_heldout_failure_summary/summary.md`: held-out failure setup and stage boundary summary.
- `experiments/dual_network/S295_center_anchored_polygon_failure_diagnostics/`: per-sample, per-component, grouped center-bin, hard-case, slot, rotated, multi-component, and worst-sample diagnostics.
- `experiments/dual_network/S296_center_anchored_polygon_matched_coverage_analysis/`: train center-bin coverage, nearest-train match, uncovered-bin tables, and matched-coverage summary.
- `experiments/dual_network/S297_center_anchored_polygon_generalization_decision/summary.md`: decision summary recommending matched-coverage resplit before model changes or larger data.

## S298-S302 COMSOL V3 polygon matched-coverage resplit artifacts

- `make_comsol_polygon_matched_coverage_resplit.py`: builds a diagnostic train/val/test resplit from the existing S254/S290 polygon V3 pack while preserving hard-case counts and improving held-out center-bin coverage.
- `smoke_test_make_comsol_polygon_matched_coverage_resplit.py`: smoke test for duplicate-free resplit assignment, distance-1 held-out coverage, and manifest writing.
- `experiments/dual_network/S298_polygon_matched_coverage_resplit_setup/summary.md`: setup summary and diagnostic boundary.
- `experiments/dual_network/S299_comsol_polygon_matched_coverage_resplit/`: matched-coverage split packages, raw metadata copies, split manifest, coverage report, and summary.
- `experiments/dual_network/S300_center_anchored_polygon_matched_coverage_probe/`: unchanged center-anchored train30 matched-coverage probe with metrics and prediction exports.
- `experiments/dual_network/S301_polygon_matched_coverage_diagnostics/`: matched-coverage per-sample results, grouped diagnostics, coverage grouping, and worst samples.
- `experiments/dual_network/S302_polygon_matched_coverage_decision/summary.md`: decision summary ruling out distance-1 coverage as sufficient and recommending y-bin localization repair.

## S303-S307 COMSOL V3 center-anchored y-bin repair artifacts

- `center_anchored_y_bin_diagnostics.py`: read-only diagnostic for center-anchored y-bin confusion, ordered y-bin error distance, offset margin, and hard-case grouping.
- `smoke_test_center_anchored_y_bin_diagnostics.py`: tempfile smoke test for y-bin diagnostic joins and outputs.
- `train_comsol_center_anchored_polygon_inverse.py`: now supports default-off `--center-y-bin-extra-loss-mode`, `--lambda-center-y-bin-extra`, `--center-y-bin-neighbor-smoothing`, and `--center-y-bin-distance-sigma`.
- `smoke_test_train_comsol_center_anchored_polygon_inverse.py`: runner smoke covering default behavior and neighbor-soft y-bin extra loss.
- `experiments/dual_network/S303_center_anchored_y_bin_failure_summary/summary.md`: stage setup and failure mechanism summary.
- `experiments/dual_network/S304_center_anchored_y_bin_diagnostics/`: y-bin confusion, y-error histogram, per-sample/per-component diagnostics, and grouped tables for the S300 reference.
- `experiments/dual_network/S305_center_anchored_y_bin_loss_repair_support/summary.md`: y-bin extra loss support summary.
- `experiments/dual_network/S306_center_anchored_y_bin_repair_quick_gate/`: current reference, neighbor-soft y, distance-soft y quick-gate metrics, prediction exports, and y-bin diagnostics.
- `experiments/dual_network/S307_center_anchored_y_bin_repair_decision/summary.md`: gate decision and next-stage recommendation.

## S308-S312 COMSOL V3 center-anchored bounded local output artifacts

- `center_anchored_local_shape_diagnostics.py`: read-only local-shape target and prediction diagnostic for center-anchored polygon runs.
- `smoke_test_center_anchored_local_shape_diagnostics.py`: tempfile smoke test for local-shape diagnostic joins and outputs.
- `train_comsol_center_anchored_polygon_inverse.py`: now supports default-off `--local-shape-output-mode raw|bounded_tanh`, `--local-shape-bound-mode fixed_grid|train_stats`, fixed local bounds, and train-stats local bounds.
- `smoke_test_train_comsol_center_anchored_polygon_inverse.py`: runner smoke covering raw, y-bin extra loss, and bounded local output code paths.
- `experiments/dual_network/S308_center_anchored_local_shape_failure_summary/summary.md`: stage setup and boundary summary.
- `experiments/dual_network/S309_center_anchored_local_shape_diagnostics/`: local-shape target ranges, grouped target stats, prediction diagnostics, and worst held-out local-shape samples for the reference.
- `experiments/dual_network/S310_center_anchored_bounded_local_output_support/summary.md`: bounded local output implementation summary.
- `experiments/dual_network/S311_center_anchored_bounded_local_output_quick_gate/`: current reference, fixed-grid bounded, train-stats bounded quick-gate runs, prediction exports, and local-shape diagnostics.
- `experiments/dual_network/S312_center_anchored_bounded_local_output_decision/summary.md`: decision summary marking bounded local output as not passing held-out repair.

## S313-S317 COMSOL V3 center-anchored local-shape conditioning artifacts

- `comsol_center_anchored_polygon_inverse_models.py`: now supports default-off local-shape conditioning modes for the local vertex head while preserving the default `none` path.
- `train_comsol_center_anchored_polygon_inverse.py`: now accepts `--local-shape-conditioning-mode` and `--local-shape-conditioning-dim`, records them in metrics/config/summary, and keeps default behavior unchanged.
- `smoke_test_comsol_center_anchored_polygon_inverse_models.py`: model smoke covering default and `center_bin_slot_type` conditioned paths.
- `smoke_test_train_comsol_center_anchored_polygon_inverse.py`: runner smoke covering CLI, metrics fields, and default-off compatibility for local-shape conditioning.
- `experiments/dual_network/S313_center_anchored_local_shape_conditioning_plan_summary/summary.md`: stage setup, mechanism, and boundary summary.
- `experiments/dual_network/S314_center_anchored_local_shape_conditioning_support/summary.md`: implementation support summary.
- `experiments/dual_network/S315_center_anchored_local_shape_conditioning_smoke/summary.md`: smoke and compile verification summary.
- `experiments/dual_network/S316_center_anchored_local_shape_conditioning_quick_gate/`: current reference and `conditioning_center_bin` quick-gate runs, local-shape diagnostics, and quick-gate metrics summary.
- `experiments/dual_network/S317_center_anchored_local_shape_conditioning_decision/summary.md`: decision summary marking simple local-shape conditioning as not passing held-out repair.

## S318-S322 COMSOL V3 center-anchored joint center/local artifacts

- `center_anchored_polygon_oracle_ablation.py`: offline teacher-forced center/local ablation for center-anchored polygon prediction exports.
- `smoke_test_center_anchored_polygon_oracle_ablation.py`: smoke test covering `pred_all`, GT center replacement, and full GT-center/local oracle recovery.
- `comsol_center_anchored_polygon_inverse_models.py`: now supports default-off `joint_center_shape_mode=soft_center_scheduled` while preserving the default `none` path.
- `train_comsol_center_anchored_polygon_inverse.py`: now accepts joint center-shape schedule options and records `joint_center_teacher_forcing_weight` in history.
- `smoke_test_comsol_center_anchored_polygon_inverse_models.py`: model smoke covering joint center-shape gradients.
- `smoke_test_train_comsol_center_anchored_polygon_inverse.py`: runner smoke covering joint center-shape CLI and schedule fields.
- `experiments/dual_network/S318_center_anchored_local_conditioning_failure_summary/summary.md`: stage setup and S313-S317 failure summary.
- `experiments/dual_network/S319_center_anchored_oracle_ablation/`: current-reference and conditioning-center-bin ablation tables and summaries.
- `experiments/dual_network/S320_joint_center_local_repair_spec/summary.md`: conditional joint repair design summary.
- `experiments/dual_network/S321_center_anchored_joint_repair_quick_gate/`: same-run reference and one `soft_center_scheduled` repair quick-gate run.
- `experiments/dual_network/S322_joint_center_local_decision/summary.md`: decision summary marking the first joint repair as not passing and recommending explicit center-local coupling next.

## S323-S327 COMSOL V3 center-anchored decoded-center coupling artifacts

- `center_anchored_center_decode_diagnostics.py`: read-only decoded-center diagnostic joining prediction exports with S299 center-anchored targets.
- `smoke_test_center_anchored_center_decode_diagnostics.py`: smoke test for hard/soft center diagnostics and summary outputs.
- `train_comsol_center_anchored_polygon_inverse.py`: now supports default-off `--center-consistency-mode`, `--lambda-center-consistency`, and `--center-consistency-smoothl1-beta`.
- `smoke_test_train_comsol_center_anchored_polygon_inverse.py`: runner smoke covering the default-off center consistency path and new prediction fields.
- `experiments/dual_network/S323_center_decode_failure_summary/summary.md`: center-decode failure and stage-boundary summary.
- `experiments/dual_network/S324_center_anchored_decoded_center_diagnostics/`: current-reference and repair-run center decode diagnostics.
- `experiments/dual_network/S325_decoded_center_coupling_support/summary.md`: implementation support summary.
- `experiments/dual_network/S326_center_anchored_decoded_center_coupling_quick_gate/`: current reference and `soft_decoded_center_consistency` quick-gate runs.
- `experiments/dual_network/S327_decoded_center_coupling_decision/summary.md`: decision summary marking loss-side decoded-center coupling as not passing.

## S328-S332 COMSOL V3 component-query center/shape artifacts

- `comsol_component_query_polygon_inverse_models.py`: independent component-query polygon inverse model with fixed learned slot queries and joint center/shape outputs.
- `train_comsol_component_query_polygon_inverse.py`: independent runner reusing center-anchored targets, hard decode, and polygon raster metrics while preserving old runner defaults.
- `smoke_test_comsol_component_query_polygon_inverse_models.py`: model smoke for forward shape, finite loss/backward, and query/head gradients.
- `smoke_test_train_comsol_component_query_polygon_inverse.py`: runner smoke for tempfile training, metrics/history/summary writing, prediction export, and no checkpoint/weight/`.npy` output.
- `experiments/dual_network/S328_component_query_route_summary/summary.md`: route motivation, model contract, and boundary summary.
- `experiments/dual_network/S329_component_query_model_runner_smoke/summary.md`: smoke verification summary.
- `experiments/dual_network/S330_component_query_polygon_overfit_gates/`: one-sample overfit run, gate metrics summary, and skipped-gate notes.
- `experiments/dual_network/S331_component_query_matched_reference/summary.md`: skipped same-run reference summary.
- `experiments/dual_network/S332_component_query_train30_decision/summary.md`: decision summary marking component-query as not yet validated.

## S333-S335 COMSOL V3 component-query raster-sensitivity artifacts

- `component_query_polygon_raster_sensitivity_diagnostics.py`: offline diagnostic for one-sample component-query polygon raster sensitivity and center/local/area/edge variants.
- `smoke_test_component_query_polygon_raster_sensitivity_diagnostics.py`: tempfile smoke test for reconstruction, variants, and output files.
- `experiments/dual_network/S333_component_query_1sample_raster_sensitivity_summary/summary.md`: setup and S330 failure summary.
- `experiments/dual_network/S334_component_query_1sample_raster_sensitivity_diagnostics/`: per-vertex errors, edge errors, mask diff summary, variant table, and diagnostic summary.
- `experiments/dual_network/S335_component_query_1sample_raster_sensitivity_decision/summary.md`: decision summary keeping 5-sample blocked and recommending center / centroid precision repair.

## S336-S340 COMSOL V3 component-query center precision artifacts

- `train_comsol_component_query_polygon_inverse.py`: now exposes default-off decoded-center and polygon-centroid auxiliary loss options for component-query one-sample precision repair.
- `train_comsol_center_anchored_polygon_inverse.py`: shared loss path now records decoded-center and polygon-centroid auxiliary losses while preserving default behavior.
- `smoke_test_train_comsol_component_query_polygon_inverse.py`: runner smoke covering default and aux-enabled component-query paths.
- `smoke_test_train_comsol_center_anchored_polygon_inverse.py`: center-anchored runner smoke covering default and aux-enabled shared loss path.
- `experiments/dual_network/S336_component_query_1sample_raster_repair_summary/summary.md`: setup and repair hypothesis.
- `experiments/dual_network/S337_component_query_center_centroid_aux_support/summary.md`: implementation support summary.
- `experiments/dual_network/S338_component_query_1sample_current_reference/`: current-reference rerun reproducing S330.
- `experiments/dual_network/S339_component_query_1sample_repair_quick_gate/`: decoded-center and polygon-centroid one-sample repair runs plus raster recheck for the best repair.
- `experiments/dual_network/S340_component_query_center_precision_decision/summary.md`: decision summary keeping 5-sample blocked.

## S341-S345 COMSOL V3 component-query boundary precision artifacts

- `train_comsol_component_query_polygon_inverse.py`: now exposes existing default-off `--lambda-area-aux` for component-query boundary precision probes.
- `smoke_test_train_comsol_component_query_polygon_inverse.py`: runner smoke covering aux-enabled component-query path including area aux CLI.
- `experiments/dual_network/S341_component_query_boundary_precision_failure_summary/summary.md`: setup and locked failure evidence.
- `experiments/dual_network/S342_component_query_boundary_precision_support/summary.md`: implementation support summary.
- `experiments/dual_network/S343_component_query_1sample_boundary_precision_quick_gate/`: current reference, center-half aux, center-half plus tiny area runs, and gate summary.
- `experiments/dual_network/S344_component_query_1sample_boundary_precision_recheck/`: raster-sensitivity recheck for all S343 runs.
- `experiments/dual_network/S345_component_query_boundary_precision_decision/summary.md`: decision summary keeping 5-sample blocked.
