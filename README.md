# 双网络变分 / weak-form 半监督支线

## 下一阶段：signal-conditioned dual-network

S48 开始把支线从 per-sample semi-supervised optimization 推进到 `signal-conditioned dual-network` 阶段。新阶段目标是训练一个共享参数模型：输入 `Bz signal + coords`，输出 `mu_pred` 和可选 `phi_pred`。

这个阶段的关键边界是推理时不使用 `mu_label` 或 `label_mask`。当前 per-sample runner 的 `BCE mask prior` 结果仍然是半监督 / diagnostic upper bound，不能直接写成主线替代结果。后续需要完成 conditional runner、batch loader、train / val / test 评估后，才可能与主线 baseline 形成可比实验。

S49 已增加 `conditional_dual_data_utils.py` 和 `smoke_test_conditional_dual_data_utils.py`，使 conditional 阶段具备从 `.npz` 读取 `signals` / `coords` / `mu_label` / `mask_label` batch 的能力。下一步建议是创建 conditional supervised training runner skeleton，先做极小 batch smoke train，再考虑正式验证。

S50 已新增 `train_conditional_dual.py` 和 `smoke_test_train_conditional_dual.py`。当前 runner 只是 supervised skeleton / smoke test：它验证 `signals + coords` 条件模型训练闭环，但没有接入 weak-form loss，也不是正式训练结果。

S53 开始做 conditional train/val probe：`train_conditional_dual.py` 已支持可选 `--eval-npz-path` 并输出 `eval_metrics.csv`。当前 20x10 小规模结果显示 train-set 拟合强，但 val IoU 仍低，因此还不能作为主线替代或泛化结论。
S54 继续扩展到 train / val / test probe：`train_conditional_dual.py` 已支持可选 `--test-npz-path` 并输出 `test_metrics.csv`。当前 20x10 结果仍表现为 train IoU 高、held-out val/test IoU 低，因此 conditional 分支还不能与主线 baseline 做正式替代比较。
S57 开始诊断 signal normalization / conditioning：`train_conditional_dual.py` 支持 `--signal-normalization none|train_zscore|per_sample_zscore`。当前结果显示简单归一化只带来小幅变化，主要瓶颈仍更像 `BzEncoder` / conditioning 架构与泛化设计。
S58 开始测试 FiLM conditioning：`conditional_dual_models.py` 和 `train_conditional_dual.py` 支持 `--conditioning-mode concat|film`。当前结果显示 FiLM 更强地拟合 train set，但没有改善 held-out val/test IoU，因此默认 conditioning 暂不切换到 FiLM。
S59 开始测试 signal encoder architecture：`conditional_dual_models.py` 和 `train_conditional_dual.py` 支持 `--encoder-type mlp|cnn`。当前结果显示 CNN encoder 提升 train 拟合，但没有稳定解决 held-out val/test 泛化，因此 encoder 结构本身还不是完整答案。
S60 开始测试 coordinate-aligned local signal features：`conditional_dual_models.py` 支持可选 `point_features`，`train_conditional_dual.py` 支持 `--point-signal-mode none|local_value|local_value_abs`。当前结果显示局部 Bz 特征没有稳定改善 held-out IoU，因此默认 conditional baseline 暂不切换到 local point features。
S61 开始测试 direct mask head：`conditional_dual_models.py` 支持可选 `predict_mask`，`train_conditional_dual.py` 支持 `--mask-head-mode mu_threshold|direct`。当前结果显示 direct mask head 明显提高 train IoU，但没有改善 held-out test IoU，因此默认 conditional baseline 仍保留 `mu_threshold`。
S63 开始测试 derived Bz signal features：`train_conditional_dual.py` 支持 `--signal-feature-mode raw|raw_abs_grad`。`raw_abs_grad` 只是对现有单条 Bz 派生 raw / abs / gradient 特征，不是 COMSOL multi-height 数据实验；当前用于判断单条 Bz 的简单派生通道是否能改善 conditional 泛化。
S64 开始支持 multi-channel / multi-height Bz interface：`conditional_dual_data_utils.py` 接受 `signals [B,L]` 和 `signals [B,C,L]`，并将多通道 signals channels-first flatten 为 `[B,C*L]`。这只是接口 skeleton，不是 COMSOL 结果，也不代表泛化效果改善。
S65 开始做 synthetic multi-height proxy probe：`build_multiheight_proxy_npz.py` 用单条 Bz 构造 raw / smooth-decay proxy channels。这不是 COMSOL 数据；下一步如果继续多高度方向，应优先考虑真实 COMSOL / multi-height Bz dataset 或转换入口。
S66 建立 COMSOL-style multi-height NPZ schema 和验证工具：`comsol_multiheight_npz_utils.py` 可检查 `signals [N,C,L]`、target 字段和推荐 metadata。当前仍没有真实 COMSOL 数据，也没有正式训练结果。

本 worktree 对应 `feature/dual-network-variational` 支线，用于探索主线之外的双网络反演方案。当前支线不替代 `main`，也不证明纯无监督 weak-form 反演已经成功。

## 支线定位

本支线探索：

- `phi-Net / mu-Net` 双网络结构；
- 使用 `phi-Net` 做变分场重构；
- 使用 `mu-Net` 做 weak-form material update；
- 在 weak-form runner 上加入半监督 `BCE mask prior`，用于诊断上界验证；
- 从单样本 prototype 推进到小规模 runner 与多分辨率实验；
- 保持与 `main` 主线隔离，不直接改主线训练流程。

关键边界：

- `BCE mask prior` 使用 `mu_label < 500` 构造 mask，因此属于半监督 / 诊断上界；
- 当前结果不能写成“纯 weak-form 无监督反演成功”；
- `label-informed centers` 是 oracle diagnostic，不是可部署方案；
- 本支线当前不建议直接合并进 `main`。

## 当前阶段结论

- `weak-form + area / soft Dice` baseline 可以工程跑通，但缺陷定位能力不足，容易出现过大的低 `mu` 区域。
- `BCE mask prior` 在 `20x10`、`40x20`、`80x40` 小规模 runner 中稳定显著优于 baseline。
- `BCE mask prior` 的有效性说明半监督 / 诊断上界有效，不代表无监督 weak-form 本身已经解决定位问题。
- 当前 `40x20` IoU 优先候选为 `temp25_lambda1`。
- 当前 `80x40` 综合候选为 `temp25_lambda3`；`temp20_lambda3` 可作为 IoU 优先参考。
- S29 显示，`80x40` 下的弱样本主要与形状细节、边界 / 窄缺陷样本、centroid 偏移和局部几何误差有关。

## 成果索引

- [支线成果索引](DUAL_NETWORK_ARTIFACT_INDEX.md)：快速定位当前文档、代码和关键实验产物。
- [复现实验说明](DUAL_NETWORK_REPRODUCE.md)：记录 20x10、40x20、80x40 的推荐配置、命令模板和结果入口。
- [跨分辨率结果报告](DUAL_NETWORK_RESULTS_REPORT.md)：汇总 20x10、40x20、80x40 的核心对比结果。
- [术语说明](DUAL_NETWORK_TERMS.md)：解释 `phi-Net`、`mu-Net`、weak-form、prior 和指标。
- [阶段总结](DUAL_NETWORK_STAGE_SUMMARY.md)：整理当前支线结论、边界和下一步建议。
- [S31 跨分辨率 SVG 图表](experiments/dual_network/S31_report_figures/)：展示 `defect_iou`、`defect_area_pred` 和 `mu_mae` 的跨分辨率对比。

## 主要文件说明

- `DUAL_NETWORK_EXPERIMENT_LOG.md`：支线实验日志，记录 S3 之后的实验过程和结论。
- `DUAL_NETWORK_STAGE_SUMMARY.md`：阶段总结，整理当前支线能力、边界和下一阶段建议。
- `DUAL_NETWORK_TERMS.md`：术语说明，解释支线中的模型、loss、prior、指标和实验编号。
- `train_dual_variational.py`：小规模 runner，对 `.npz` 中多个 sample 独立运行双网络 weak-form loop，并输出 `metrics.csv`。
- `minimal_dual_single_sample_loop.py`：单样本 prototype，用于快速验证单样本闭环和诊断输出。
- `dual_network_models.py`：定义 `PhiNet` 和 `MuNet`。
- `dual_network_losses.py`：定义 `energy_loss`、`data_loss`、`tv_loss`、`weak_form_loss`、`generate_compact_support_test_grads` 等。
- `dual_network_data_utils.py`：读取 `.npz` 数据、构造 `coords`、构造 probe 坐标和单样本输入。
- `experiments/dual_network/`：支线实验记录、summary、metrics 和代表性图像。

## 运行示例

以下命令只是 runner 用法示例，不要求直接运行：

```powershell
python train_dual_variational.py --npz-path path/to/train.npz --output-dir experiments/dual_network/example_run --sample-indices 0,1,2 --outer-steps 30 --phi-steps 30 --mu-steps 30 --test-radius 5.0 --center-mode three --lambda-area-prior 1.0 --lambda-mask-prior 1.0 --lambda-mask-bce-prior 3.0 --mask-prior-temperature 25.0
```

## 文档语言规范

- 顶层支线文档正文默认使用中文。
- 文件名、参数名、指标名、命令和代码标识保留英文。
- 历史 `experiments/*/summary.md` 是实验产物，不强制逐个翻译。
- 从 S29 之后新增 `summary.md` 尽量使用中文正文。

## 边界说明

- 不要把 `BCE mask prior` 结果写成无监督成功。
- 不要把 `label-informed centers` 写成可部署方法。
- 不要把本支线直接同步或合并进 `main`。
- 不建议继续盲目扫描 `test_radius`、`center_mode` 或 `area prior`。
- 若继续推进，应优先整理阶段性报告，或针对 S29 的弱样本做定向失败分析。
S67 增加 `convert_comsol_multiheight_csv_to_npz.py`，用于把 COMSOL-style long CSV 转换为 multi-channel NPZ。当前验证只使用 mock CSV data，不代表真实 COMSOL export 或模型效果。
S68 增加 `COMSOL_PILOT_DATA_REQUEST.md`，用于向 COMSOL MCP / COMSOL 项目请求 5-10 个真实 multi-height Bz pilot samples。当前仍未提交真实 COMSOL 数据。
S69 增加 `smoke_test_comsol_pilot_handoff_end_to_end.py`，用 tempfile 模拟 COMSOL pilot export，并验证 CSV -> NPZ -> validator -> conditional batch -> model forward 链路。真实 pilot 仍需在 COMSOL MCP 侧生成。
S70 增加 `COMSOL_MCP_PILOT_PROMPT.md`，提供可直接交给 COMSOL MCP 的 pilot prompt。当前支线已准备好真实 COMSOL pilot 数据接入前的 converter、validator 和 dry-run 链路，但仍没有真实 COMSOL 数据。
S71 已接入第一批真实 COMSOL pilot：`signals [5,3,200]` 经 S67 converter 转换为支线可读 NPZ，并通过 validator、conditional batch flatten `[B,600]` 和 model forward 检查。该 pilot 固定仿体，只改磁性参数，因此主要验证接口链路；后续需要几何变化数据才能判断 shape generalization。
S72 使用 S71 converted NPZ 做了 5-sample sanity training，确认真实 COMSOL pilot 可以进入 `train_conditional_dual.py` 并输出 metrics；这不是正式训练，也不代表泛化能力。
S73 新增 `COMSOL_GEOMETRY_VARIATION_DATA_REQUEST.md`，规定下一批 COMSOL 数据必须变化 defect center、size、depth、shape 和 permeability / mu，并建议先做 train / val / test split，用于后续真正的 conditional shape generalization probe。


## COMSOL geometry-variation conditional probe

S74/S75/S76 已完成第一批真实 COMSOL geometry-variation multi-height Bz 数据接入和第一轮 conditional train / val / test probe。S74 将 `comsol_geometry_variation_exports/` 中的 train / val / test long CSV + target NPZ 转换为支线可读 multi-channel NPZ；S75 使用 `train_point_subsample=4096` 运行 `medium_multichannel` 和 `big_multichannel` 两组 supervised probe；S76 总结结果和下一步。

当前 S75 held-out IoU 约为 0.40，明显高于早期 synthetic single-Bz conditional 阶段约 0.1 左右的 held-out IoU，说明真实 multi-height Bz 数据更有潜力。但这仍是 pilot 级结果：train samples 只有 50，val/test 各 10，`defect_type` 固定为 `ellipsoid`，未覆盖旋转角、边界不规则度或多缺陷类型。后续不应直接声称主线替代成功，应继续扩大 COMSOL geometry 数据并检查 target/mask、loss balance 和 validation-aware selection。


## COMSOL target/mask and train-fit diagnostics

S77-S80 完成了真实 COMSOL geometry 数据的 target/mask 与 train-fit 诊断。S77 证明 `mu_maps < 500` 与 provided `masks` 完全一致；S78 增加 `mask_source=mu_threshold|masks`，并确认 provided masks 没有改善 train / val / test；S79 测试 longer steps、larger point subsample 和更高 BCE 权重，均未显著提升 train fit。

当前结论是：target/mask source 不是主要瓶颈；简单增加训练步数、采样点或 BCE 权重也不是主要解决方案。下一步应保留默认 `mask_source=mu_threshold`，重点检查 COMSOL 数据的 target / loss / model 表达，并扩大 geometry variation 数据的数量和形状多样性。


## COMSOL output head diagnostics and V2 data request

S81-S83 ????? COMSOL conditional ?????????????????S81 ? S74 geometry-variation ????? `mu_threshold_reference`?`direct_mu0` ? `direct_mu1e-5`?direct head ?????? held-out IoU??? `mu_mse` ????? `mu` ?????????? mask IoU ???

S82 ??????????????? data size/diversity???? model/conditioning?target/mask ?? S77/S78 ?????output head ???????????

S83 ?? `COMSOL_GEOMETRY_VARIATION_DATA_REQUEST_V2.md`?????? COMSOL ????? train 200 / val 50 / test 50???? fallback ? train 100 / val 20 / test 20?????? defect type?rotation angle ? boundary irregularity?

## COMSOL geometry V2 ingest and probe

S84/S85/S86 已完成第二批真实 COMSOL geometry V2 fallback 数据接入和第一轮 conditional probe。S84 将 `comsol_geometry_variation_v2_exports/` 的 train / val / test long CSV + target NPZ 转为支线可读 multi-channel NPZ，signals shape 为 train `[100,3,200]`、val `[20,3,200]`、test `[20,3,200]`，flattened signal length 为 `600`。

S85 的 `big_multichannel_v2` 是当前 V2 较好配置，train / val / test IoU 为 `3.023806e-01` / `2.593440e-01` / `2.768323e-01`，低于 V1 S75 的 held-out IoU 约 `0.40`。因此 V2 fallback 暂未显示更好的 val/test 潜力；下一步应先诊断 V2 target/mask、signal 语义、lift-off 定义和 runner/loss 适配，而不是直接继续扩大 V2 样本数。

V2 当前边界：fallback 规模 train=100 / val=20 / test=20；包含 `rectangular_notch` / `rotated_rect` multi_defect 和 rotation variation；不包含 `ellipsoid`；`boundary_irregularity` 是 proxy；magnetic parameters 固定。

## COMSOL V2 target and signal diagnostics

S87-S90 已完成 V2 target/signal 诊断。S87 显示 V1/V2 的 `mu_maps < 500` 与 provided `masks > 0.5` 完全一致，因此 target/mask 定义不是 V2 低于 V1 的主因。但 V2 label area 明显更小：V2 train mean label area ratio 为 `5.355850e-02`，V1 train 为 `1.172090e-01`，V2 只有 V1 的 `45.7%`。

S88 显示 V2 signal scale 与 V1 有明显差异，V2 train mean_peak_abs_signal 是 V1 的 `11.747x`，但 V2 offset/peak 仅 `0.041`，lift-off 通道总体符合高度增加后衰减的预期。由于 S85 已使用 `per_sample_zscore`，S89 未再执行 center-only 训练。

当前判断是：V2 低于 V1 更可能来自 label area 更小、multi_defect / non-ellipsoid 任务更难，以及当前 runner/loss 对 small-label multi-component target 不适配；下一步不应直接扩大 V2 数据，而应先做 small-label / multi_defect 目标的 loss 和 sampling 适配。
## COMSOL V2 small-label adaptation

S91-S94 针对 V2 label area 更小、multi_defect / non-ellipsoid 更难的问题，扩展并测试了 conditional runner 的 small-label 适配能力。

新增 runner 参数包括：

- `--mask-bce-mode bce|pos_weighted_bce|focal_bce`
- `--pos-weight`
- `--focal-gamma`
- `--focal-alpha`
- `--point-sampling-mode random|positive_balanced`
- `--positive-fraction`

S93 在 V2 数据上测试 `balanced_bce`、`balanced_pos_weight5` 和 `balanced_focal`。三组都退化为零面积预测，未改善 S85 `big_multichannel_v2` baseline。因此这些配置目前只作为诊断能力保留，不作为 V2 默认训练策略。

下一步更适合回到 S85 baseline，测试 direct mask、area calibration、boundary-aware objective、curriculum 数据或模型/conditioning 调整。
## COMSOL V1-to-V2 curriculum bridge

S95-S98 针对 S93 small-label adaptation 失败后继续诊断 V2 训练动态。

`train_conditional_dual.py` 新增：

- `--history-interval`：输出 `training_history.csv`；
- `--pretrain-npz-path`；
- `--pretrain-sample-indices`；
- `--pretrain-steps`。

S97 测试了 V2-only reproduce 和 V1 pretrain -> V2 finetune。两组最终都退化为全背景预测，train / val / test IoU 均为 0。V1 pretrain 阶段本身能正常拟合，但进入 V2 finetune 后仍塌缩。

当前判断是：V2 问题不是简单 warm start 或 V1-like pretrain 能解决的初始化问题；下一步应优先处理 positive area / mask output dynamics，例如 area calibration、positive area prior、direct mask head、boundary-aware objective，或准备 mixed curriculum 数据。

## COMSOL V2 background-collapse suppression

S99-S102 针对 V2 conditional training 的全背景塌缩做了 area dynamics 诊断。`train_conditional_dual.py` 新增 `--area-loss-mode none|batch_ratio_mse|foreground_floor`、`--lambda-area-loss` 和 `--foreground-floor-ratio`，并在 `training_history.csv` 中记录 `area_loss`、`pred_area_soft_mean` 和 `true_area_mean`。

S101 在 V2 train=100 / val=20 / test=20 数据上比较了 `v2_baseline_with_history`、`area_ratio_mse` 和 `foreground_floor`。三组 train / val / test IoU 均为 0，hard `defect_area_pred` 均为 0。`area_ratio_mse` 可以降低连续 `mu` 误差，但没有恢复 hard foreground。

当前结论是：简单 area ratio / foreground floor 不能单独解决 V2 全背景塌缩。下一步更适合测试 direct mask + area loss、threshold margin 诊断、boundary/localization loss，或 V1-like -> intermediate -> V2-like staged curriculum。

## COMSOL V2 hard-threshold margin objective

S103-S106 针对 S101 中 soft foreground 非零但 hard mask 仍为 0 的问题，增加了 hard `mu_threshold` crossing 诊断和 threshold-margin loss。

S103 显示 S101 三组都存在 `soft-hard mismatch` 和 `no-threshold-crossing`：最后 history 的 `min_mu` 都高于 `500`。S104 因此在 `train_conditional_dual.py` 中新增 `--threshold-margin-mode none|positive_hinge|bidirectional_hinge`、`--lambda-threshold-margin`、`--positive-mu-margin` 和 `--negative-mu-margin`。

S105 结果显示 positive-only margin 能恢复 hard foreground，但会变成全前景；`bidirectional_margin_lambda1` 恢复了非零 IoU 并避免全前景，但仍低于 S85 baseline。当前下一步应转向 direct mask + margin、boundary/localization loss，或 staged curriculum。

## COMSOL V2 quick diagnostic gates

S107-S111 将 V2 localization objective 的 full-run 搜索提前收束。S109 已完成 `bidir_margin_val_select`、`bidir_margin_area_ratio` 和 `bidir_margin_floor`，但没有超过 S85 `big_multichannel_v2` baseline；`direct_mask_area_ratio` 未运行，因阶段策略切换而停止。

后续 COMSOL V2 objective / output path 不再直接进入 full train=100 / val=20 / test=20 长实验。新方案必须先通过 `COMSOL_V2_QUICK_DIAGNOSTIC_GATES.md`：Gate 1 是 5-sample train-overfit，Gate 2 是 20-train / 5-val mini generalization，只有前两级通过后才允许 full V2 probe。

## COMSOL parametric inverse route

S112 开始测试 parametric inverse route：利用 V2 `defect_params`，先从 multi-height Bz signal 预测 component-level 几何参数，再由预测参数 rasterize 成 mask 评估 IoU / Dice。该路线用于降低输出空间维度，避免继续在 dense sparse mask 上盲目调 loss。当前阶段只做 skeleton、smoke 和 small train probe，不保存权重，也不作为主线替代结论。

S113-S116 已完成首个 parametric inverse skeleton 和 V2 probe。S113 从 `source_component_json` 构造 `max_components=3` 的 component targets，`type_vocab=rectangular_notch, rotated_rect`。S115 的 parametric probe 在 val / test 上得到 rasterized mask IoU `0.369908` / `0.424462`，说明该路线已有非平凡信号；当前主要瓶颈变为 rotation / type 泛化、rasterization 近似和 signal encoder 表达能力。

S117-S120 完成了 parametric oracle/refinement decision。S117 用 GT parametric targets rasterize 回 target masks，train / val / test oracle IoU 为 `0.722997` / `0.723288` / `0.716584`，通过 `0.70` gate。S118 增加 angle sin/cos、train-stat continuous normalization 和 inverse-frequency type weighting。S119 refined MLP 的 val / test mask IoU 为 `0.325765` / `0.388509`，没有改善 S115，因此后续应继续 parametric route，但优先改 signal encoder、component head、type/rotation loss 分解和 rasterizer semantics，而不是只延长当前 MLP 训练。

S121-S125 已完成首轮 parametric component-head / encoder 诊断。S121 确认主要 held-out gap 来自 type / rotation / geometry generalization，而不是 presence 或 oracle rasterizer 上限。S122 增加 `encoder_type=mlp|cnn1d|cnn1d_attention` 和 `head_mode=shared|component_specific`。S123/S124 显示 CNN1D/component-specific 在 quick gate 中略改善 type / rotation，但没有超过 S115 raw MLP 的 val/test mask IoU；longer run 还会退化。因此当前最佳仍是 S115 raw parametric baseline，下一步应转向 per-sample prediction export、grouped diagnostics、slot decoder / set prediction 或 forward consistency。

S126-S130 已完成 per-sample prediction diagnostics 和 set-matching probe。S126 为 parametric runner 增加 `--export-predictions`，导出每个 sample / component 的 prediction、target、type、rotation 和 mask IoU / oracle gap。S127 grouped diagnostics 显示主要问题仍是 held-out type / rotation / geometry generalization，slot 1 相对较弱但不是唯一瓶颈。S128/S129 测试 `component_matching_mode=permutation_min`，结果 val / test mask IoU 降到 `0.178724` / `0.246229`，低于 fixed S115 baseline 的 `0.369908` / `0.424462`。因此 loss-side permutation matching 不作为当前默认路线；下一步更适合 forward consistency / differentiable rasterization、geometry-aware rotation/type objective，或更明确的 slot/query decoder。

S131 已确认 set matching 不是主方向，下一步转向 differentiable raster mask supervision：用可微 soft rasterization 把预测几何参数转为 soft mask，让 mask-level loss 直接反传到 geometry parameters。该方向仍是 parametric route 内的 pilot，不等同 COMSOL forward consistency。

S132-S133 已实现 PyTorch differentiable soft rasterizer，并将 raster BCE / Dice loss 可选接入 parametric inverse runner。默认 raster loss 关闭，S115 raw MLP baseline 兼容性保持不变。该 raster supervision 只约束 mask-level geometry，不是 COMSOL field forward consistency；是否改善 held-out mask IoU 由后续 S134/S135 判断。

S134-S135 已完成 differentiable raster supervision quick gate。`raster_dice1` 将 test mask IoU 从 `0.424462` 小幅提升到 `0.438508`，但 val mask IoU 从 `0.369908` 降到 `0.352389`；其他 raster loss 配置也未稳定超过 parameter-only baseline。因此当前最佳稳定配置仍是 S115 raw MLP / shared head / fixed-order baseline。下一步若继续 raster route，应优先测试两阶段 parameter-only prefit + raster fine-tune、validation-aware selection 或 forward consistency，而不是继续盲扫 raster loss 权重。

S136 已将 raster supervision 收束为 fine-tune 方向：从头加入 raster loss 不作为默认，下一步测试 parameter prefit 后再启用 raster mask loss，并用 validation-aware endpoint selection 控制 fine-tune 退化风险。

S137-S139 已完成 two-stage raster fine-tune probe。`train_comsol_parametric_inverse.py` 现在支持 `--raster-loss-start-step` 和 validation-aware endpoint selection。S138 中 `param_only_val_select` 提升 val mask IoU 但 test 低于 S115；`two_stage_raster_dice` 提升 train mask IoU，但 val/test 仍未超过 S115 / S134 parameter-only baseline；`two_stage_raster_bce_dice` 明显劣化。因此 two-stage raster fine-tune 暂不作为默认。Parametric route 继续，但下一步更适合 forward consistency / physics feature extraction，或 very short post-selection raster fine-tune，而不是继续盲扫 raster BCE / Dice 权重。

S140 已将下一阶段明确转向 physics-based MFL signal features：显式提取 multi-height Bz 的 peak、width、energy、lift-off decay ratio 和 inter-channel correlation，并测试 physics features / raw+features 是否能改善 parametric inverse 的 held-out type、rotation 和 mask IoU。

S141 已新增 `comsol_mfl_physics_features.py`，在 V2 train / val / test 上生成 `physics_features [N,58]`，包括峰值、峰位、峰宽、能量、lift-off 衰减比例和通道相关性。后续 S142/S143 将比较 raw signal、features-only 和 raw+features fusion。

S142 已将 physics features 接入 parametric inverse runner：`feature_fusion_mode=features_only` 只使用 features，`feature_fusion_mode=concat_latent` 融合 raw signal latent 与 feature latent。feature normalization 只使用 train stats，默认 `none` 保持旧 raw signal 行为。

S143-S144 已完成 physics feature fusion quick gate。`physics_features_only` 能强拟合 train，但 val/test mask IoU 明显低于 raw baseline；`raw_plus_physics_features` 对 val type / rotation 有局部改善，但没有改善 held-out mask IoU。当前最佳仍是 S115 / S143 raw MLP / shared head / fixed-order baseline。下一步更适合 forward consistency / learned forward surrogate，或把 physics features 作为 auxiliary prediction / regularization，而不是直接 concat 输入。

S145 已收束 physics feature fusion 阶段：direct feature concat 不作为当前默认路线，raw signal MLP baseline 仍是最稳配置。下一步转向 learned forward surrogate / forward consistency，即学习 geometry parameters -> multi-height Bz signal 的轻量 surrogate，并用它诊断 inverse model 预测的 geometry 是否能解释输入 Bz。

S146 已新增 learned forward surrogate skeleton：`geometry_vector` 将 fixed-order component 的 presence、type 和 continuous parameters 展平，MLP surrogate 输出 multi-height Bz signal。该路线使用 train-only signal z-score，并只输出 metrics / summary，不保存 surrogate 权重；S147 会先验证 surrogate quality，再决定是否进入 forward-consistency inverse training。

S147 forward surrogate quality gate 已通过，val/test `signal_corr` 均超过 `0.80` 且 `signal_nrmse_raw < 1.0`。S148 因此新增 in-memory forward-consistency inverse runner：先训练并冻结 learned surrogate，再用 signal reconstruction residual 约束 inverse geometry；surrogate 和 inverse 权重均不保存。

S149-S150 已完成 learned forward consistency probe。Forward surrogate 可作为 diagnostic referee，但简单 forward residual objective 没有超过 S115 / S143 raw MLP baseline：`lambda=0.1` 对 rotation 有局部改善但降低 mask IoU，`lambda=1.0` 过强并明显退化。当前最佳 parametric 配置仍是 raw MLP / shared head / fixed-order parameter-only baseline；下一步不建议盲扫 consistency lambda。

S151 将 learned forward surrogate 从默认 training objective 暂时转为 residual diagnostic：先检查 residual 是否能区分 type / rotation / axis 几何错误，再决定是否继续 forward consistency。同时下一步直接测试 type / rotation targeted supervision。

S152 residual diagnostic 显示 forward residual 对 rotation 错误很敏感、对 type 错误中等敏感，但对 axis scaling 基本不敏感，且 predicted residual 与 mask IoU 相关性有限。因此当前不把 forward residual 作为默认 loss，转向更直接的 type / rotation targeted supervision。

S153-S155 已完成 type / rotation targeted supervision 诊断。`type_extra` 没有改善 type accuracy；`rotation_extra` 改善 val mask IoU 但 test 低于 S115；组合 loss test 退化。当前最佳稳定配置仍是 S115 / S143 raw MLP / shared head / fixed-order parameter-only baseline。下一步更适合 type/rotation-balanced data、type-specific heads / auxiliary classifier 或更强 target representation，而不是继续调同类 loss 权重。

S156-S157 将下一步切换为 parameter-level oracle ablation：不再继续盲扫 type / rotation loss，而是从已有 S126 per-component predictions 出发，逐项替换 GT type、rotation、center、axis、depth 和 continuous，再用 hard rasterizer 直接评估最终 mask IoU 变化。该阶段不运行新训练，用于判断哪个参数误差真正限制 parametric route。

S158-S159 已完成 oracle ablation。`pred_all` 复现 S115 baseline，`gt_all` 对齐 S117 oracle；最大的稳定提升来自 `gt_center`，val / test mask IoU 提升到 `0.714872` / `0.722920`。`gt_rotation` 没有改善 val/test，`gt_type` 和 `gt_depth` 在当前 hard rasterizer 下不直接改变 mask。因此当前主要瓶颈更新为 component center localization，下一步更适合 center-targeted representation / loss / auxiliary head，而不是继续盲扫 type、rotation 或 forward-consistency loss。

S160-S165 已完成 center-localization 诊断与 quick fix。S161 显示 val/test center error 与 mask IoU 强负相关，`center_l2_grid_mae` 为 `8.017750` / `6.998191`；S162 增加 `lambda_center_grid` 与 `lambda_center_axis_relative`，默认保持旧行为；S163/S164 显示 `lambda_center_grid=0.1` 将 3000-step val/test mask IoU 提升到 `0.469423` / `0.498874`。当前最佳 parametric 配置更新为 raw MLP / shared head / fixed-order + center grid loss；下一步优先做 center-bin classification + offset 或 signal-to-center auxiliary head，而不是继续盲扫 type / rotation / forward / raster loss。
S166-S170 已完成 center-grid stability validation。`train_comsol_parametric_inverse.py` 支持 `--seed` 并记录到 metrics / summary；S168 复用 S164 `existing_unrecorded` 并新增 seed1/seed2。三次 center-grid runs 的 val/test IoU 分别为 `0.469423/0.498874`、`0.485716/0.505590`、`0.446966/0.503713`，全部高于 historical param-only baseline。当前 COMSOL parametric route candidate 更新为 raw MLP / shared head / fixed-order + `lambda_center_grid=0.1`；后续不应继续简单 lambda sweep，而应围绕 center representation 做结构性改进。

S171-S175 completed documentation-only consolidation for the current COMSOL parametric route candidate. The branch candidate is raw MLP / shared head / fixed-order + `lambda_center_grid=0.1`, with `lambda_center_axis_relative=0.0`, no raster loss, no forward consistency, and no validation-aware endpoint selection. This is only the current candidate on `feature/dual-network-variational`; it is not a main baseline replacement. The next recommended route is `center-bin classification + offset`.
S176-S180 implemented and probed center-bin + offset localization. The best quick-gate/full-confirm configuration was `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, and `lambda_center_grid=0.1`; S179 reached val/test mask IoU `0.542935` / `0.581320`. This is promising but not yet promoted over the S170 candidate because it still needs multi-seed stability validation.
S181-S185 completed that stability validation. Reusing S179 seed1 and adding seed2/seed3, `center_bin_offset_plus_grid` produced test IoU `0.581320` / `0.575504` / `0.578738`, all above the S170 center-grid range, with lower held-out `center_grid_mae`. The current COMSOL parametric route candidate on `feature/dual-network-variational` is now raw MLP / shared head / fixed-order + `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, and `lambda_center_grid=0.1`. This remains branch-local and is not a main baseline replacement; seed2/seed3 have lower val IoU than S179 seed1, so center-bin validation should continue observing val stability.
S186-S188 consolidated that candidate and diagnosed the remaining center-bin errors without running new training. S187 shows test is stable because test `center_grid_mae` stays near `2.72-2.93`, while val is more variable because seed2/seed3 val `center_grid_mae` rises above `6.0`. The likely residual bottleneck is x-bin stability first, with y-bin secondary. The next unique recommended route is `signal-to-center auxiliary head`, not more center lambda tuning or a return to type/rotation, raster, or forward-consistency sweeps.

S189-S193 implemented an optional signal-to-center auxiliary head and quick-gated it against a same-round S185 current-candidate reference. The reference reproduced a strong 1500-step result with val/test IoU `0.546311` / `0.586546`; the auxiliary variants did not exceed it (`0.516648` / `0.567790` and `0.542723` / `0.580217`). S192 full confirm was skipped, and the current branch COMSOL parametric candidate remains the S185 `center_bin_offset_plus_grid` configuration. The auxiliary head is available for diagnostics but is not promoted.

S194-S197 diagnose the remaining center-bin failures from existing S191 prediction exports. The current candidate reference shows x-bin errors dominate y-bin errors on held-out splits: val/test x wrong rates are `0.200000` / `0.133333`, versus y wrong rates `0.083333` / `0.033333`. The auxiliary head did not help because it failed to improve the final decoded center-bin behavior used for rasterization. The next recommended route is center-x-bin focused calibration / hard-sample refinement of the main center-bin output, while keeping S185 `center_bin_offset_plus_grid` as the current branch candidate.

S198-S202 tested that recommendation as a narrow main center-bin CE reweighting gate. `train_comsol_parametric_inverse.py` now supports optional `--center-bin-x-weight`, `--center-bin-y-weight`, and `--center-bin-slot-weights` knobs for the main center-bin loss while preserving the default behavior. S200 used a same-round 1500-step reference with prediction exports; the reference reproduced val/test IoU `0.546311` / `0.586546`, while `x_bin_weighted` reached `0.545284` / `0.555791` and `x_bin_slot_weighted` reached `0.518272` / `0.543191`. The quick gate failed, S201 was skipped, and the current branch candidate remains S185 `center_bin_offset_plus_grid`. The next route should inspect low-IoU samples where bins are correct but offset / decoded center / geometry interaction still fails, rather than continuing simple x-bin or slot-weight sweeps.

S203-S206 freezes the S185 `center_bin_offset_plus_grid` candidate and converts the S198-S202 negative result into a hard-sample / data-design package. S204 uses existing S200 exports only: 23 held-out hard sample keys are split into a mixed taxonomy, including `x_bin_wrong=12`, `both_bins_wrong=3`, `bins_correct_center_or_offset_bad=5`, `geometry_or_type_interaction=2`, and `y_bin_wrong=1`. S205 drafts `COMSOL_V3_HARD_CASE_DATA_REQUEST.md` for a small V2-compatible hard-case pack, not a broad V3 expansion or a training claim. S206 recommends using that hard-case request plus bins-correct low-IoU diagnostics before any new training; the branch candidate remains unchanged.

S207 refines the V3 hard-case pack request into a concrete preflight spec. The default request is a small `60/20/20` train/val/test pilot with hard-case labels split across `x_bin_wrong_like`, `both_bins_wrong_like`, `bins_correct_center_or_offset_bad`, `geometry_or_type_interaction`, and `rare_y_bin_wrong`; if COMSOL generation cost is high, the fallback is `30/10/10`. The pack must remain V2-compatible (`signals_multiheight.csv`, `targets.npz`, `defect_params.csv`, `README.md`) and should pass an ingest gate before any training. S185 `center_bin_offset_plus_grid` remains the frozen branch candidate.

S208-S211 completed the ingest gate for the real COMSOL V3 hard-case fallback pack. The ingested train/val/test shapes are `[30,3,200]`, `[10,3,200]`, and `[10,3,200]`; all signal values are finite, `masks` exactly match `mu_maps < 500`, and hard-case labels cover `x_bin_wrong_like`, `both_bins_wrong_like`, `bins_correct_center_or_offset_bad`, `geometry_or_type_interaction`, and `rare_y_bin_wrong`. Parametric targets use `center_x`, `center_y`, `axis_x`, `axis_y`, `depth_or_shape_param`, and `rotation_angle`, and oracle rasterization reaches train/val/test IoU `1.000000` / `1.000000` / `1.000000`. The pack can proceed to branch-local candidate evaluation, but it is still a fallback pilot with single rectangular Block solves, not true rotated or multi-component COMSOL geometry.

S212-S216 evaluated the S185/S181 `center_bin_offset_plus_grid` branch candidate on the V3 hard-case fallback pilot. V2-trained zero-shot evaluation was blocked because V3 center coordinates use `[0,4500]` / `[0,3000]`, outside the V2 meter-scale center-bin grid. Same-grid V3 train -> V3 val/test training was also weak: the center-bin candidate reached val/test IoU `0.046905` / `0.044968`, and the continuous param-only reference reached `0.078177` / `0.036448`. The current candidate is therefore not validated on this V3 fallback pack. The next step is geometry-coordinate harmonization or explicit V3-to-V2 unit conversion before another ingest/oracle and candidate-evaluation run.

S217-S221 normalizes the V3 hard-case geometry convention. V3 raw `x/y` are COMSOL model coordinates `[0,4500]` / `[0,3000]`; S218 maps them to the V2-compatible meter convention `[-0.04,0.04]` / `[-0.01,0.01]`, with matching transforms for `defect_center_x/y` and `axis_x/y`. `signals`, `mu_maps`, and `masks` are unchanged, and depth/z is deliberately retained raw. Normalized oracle rasterization remains train/val/test IoU `1.000000` / `1.000000` / `1.000000`; a 5-step runability gate confirms the previous center-bin range error is gone. This stage only fixes coordinate convention and runability; V3 candidate performance evaluation is deferred to the next stage.

S222-S226 evaluates the current S185 `center_bin_offset_plus_grid` branch candidate on the normalized V3 fallback pilot. Zero-shot V2-train to normalized-V3 val/test now runs but reaches only `0.002348` / `0.012360` mask IoU; normalized V3 train quick probe is also weak, with candidate train/val/test IoU `0.019538` / `0.047127` / `0.044771`. The current candidate remains the V2-style branch candidate, but it is not validated on this V3 fallback pilot. Next step: generate a larger real COMSOL V3 hard-case pack with true rotated and multi-component geometry coverage before further model/loss changes.

S227-S231 stops that data-expansion path and first checks V3 train learnability. Target/mask/bbox/center-bin alignment is sane, but every normalized V3 sample triggers the runner `std < 1e-8` signal floor; train signal std is only `4.734403e-10` to `9.312454e-09`. Tiny-overfit training was therefore skipped. The next step is to inspect COMSOL V3 Bz signal export semantics, probe height, field expression, source scaling, lift-off extraction, and the runner normalization floor before generating more V3 data or changing the model.

S232 validates the repaired COMSOL V3 Bz signal export route with a 3-sample real COMSOL mini-smoke. The repaired route uses a near-defect probe and anomaly / delta-Bz signal instead of raw near-constant absolute Bz. The three samples cover `x_bin_wrong_like`, `bins_correct_center_or_offset_bad`, and `rare_y_bin_wrong`; all sample/channel signals have std above `1e-8`, peak-to-peak above `1e-8`, complete `x_index`, finite values, and `masks == (mu_maps < 500)`. The converted smoke shape is `[3,3,200]`. This remains a smoke validation, not a fallback pack or training result; the next step is to generate a repaired V3 hard-case fallback pack.

S233-S236 ingests the repaired V3 hard-case fallback pack generated with per-sample fresh COMSOL models and repaired near-defect `mfnc.redBz` signals. The ingested train/val/test shapes are `[30,3,200]`, `[10,3,200]`, and `[10,3,200]`; hard-case distributions re-counted from `defect_params.csv` are train `10/5/7/5/3`, val `3/2/2/2/1`, and test `3/2/2/2/1`. Signal std ranges are `1.678613e-06`-`3.162381e-06`, `1.943240e-06`-`2.826707e-06`, and `2.101787e-06`-`2.876971e-06`; `masks == (mu_maps < 500)` mismatch is `0`. Parametric targets use `center_x`, `center_y`, `axis_x`, `axis_y`, `depth_or_shape_param`, and `rotation_angle`, and oracle rasterization reaches train/val/test IoU `1.000000` / `1.000000` / `1.000000`. The repaired V3 ingest gate passes; next step is branch-local repaired V3 candidate evaluation, still bounded to single unrotated Block fallback geometry.

S237-S241 evaluates the current S185 `center_bin_offset_plus_grid` branch candidate on the repaired V3 fallback pilot. V2-train zero-shot to repaired-V3 val/test still fails before metrics because repaired V3 remains in raw COMSOL coordinates and trips the train-grid center-bin range check. Same-grid repaired V3 training is learnable on train: candidate train IoU reaches `0.998851`, but held-out val/test are only `0.052874` / `0.197143`; param-only reaches `0.986927` train and `0.000000` / `0.157851` val/test. The repaired signal route is therefore usable, but this `30/10/10` fallback pack is too small and split-sensitive for a stable candidate decision. The next recommendation is a larger repaired V3 hard-case pack, not a model-structure change.

S242-S246 normalizes the repaired V3 geometry convention and reruns the branch-local evaluation. Raw COMSOL `x/y` coordinates `[0,4500]` / `[0,3000]` are mapped to V2-compatible `[-0.04,0.04]` / `[-0.01,0.01]`, with matching center and axis transforms. Normalized oracle rasterization remains perfect at train/val/test IoU `1.000000` / `1.000000` / `1.000000`. V2-train zero-shot now runs but remains weak (`0.007616` / `0.005248` val/test IoU). Training on normalized repaired V3 fits train (`1.000000` IoU) but not held-out val/test (`0.055172` / `0.188341`). The S185 center-bin candidate remains branch-local; next step is a larger repaired V3 hard-case pack before mixed V2+V3 training or candidate promotion.

S247-S253 starts the COMSOL V3 polygon / corner-point geometry route. The old `center + axis + rotation` schema is insufficient for true rotated V3 geometry after non-uniform raw-to-V2 coordinate normalization, so this stage adds fixed four-corner polygon targets and a hard polygon rasterizer. A real 3-sample COMSOL smoke with true rotation and true multi-component Union passed the polygon oracle gate with per-sample IoU `1.000000`. No training was run, no larger pack was generated, and the S185 `center_bin_offset_plus_grid` candidate remains the current V2-style branch candidate.

S254-S258 ingests the polygon-compatible repaired V3 hard-case pack into the branch experiment tree. Converted train/val/test shapes are `[30,3,200]`, `[10,3,200]`, and `[10,3,200]`; repaired Bz signals are finite and non-near-constant, hard-case distributions match the requested `10/5/7/5/3`, `3/2/2/2/1`, and `3/2/2/2/1`, and polygon oracle train/val/test mean and min IoU are all `1.000000`. This validates the pack for the next polygon inverse route stage. It does not train a model, replace the S185/S181 center-bin candidate, or become a main baseline replacement.

S259-S263 adds the first independent polygon inverse model and runner. The model predicts fixed-slot presence/type/four-corner vertices from multi-height Bz, and hard polygon rasterization is used only for evaluation. Smoke tests pass, but the one-sample overfit gate stops the stage: train sample `0` reaches presence/type accuracy `1.000000` and vertex MAE `4.207401e-05`, while hard polygon mask IoU is `0.883178`, below the `0.90` stop threshold. The next step is vertex-to-raster sensitivity diagnosis before scaling polygon inverse training.

S264-S268 repairs that one-sample polygon inverse gate. S265 shows the S262 failure was hard-raster sensitivity: target-vertex oracle IoU remains `1.000000`, but sub-cell vertex drift adds `25` false-positive pixels and expands area from `189` to `214`. S266 adds default-off grid-space vertex loss plus area/edge auxiliary loss support. The first repair run, `longer_overfit`, reaches train IoU `1.000000`, vertex MAE `7.786439e-07`, and pred/target area `189` / `189`, so the polygon route can resume the staged 5-sample gate next. This still does not replace the S185/S181 center-bin branch candidate or the main baseline.

S269-S273 runs the polygon inverse 5-sample overfit gate only. The subset covers all five V3 polygon hard-case labels using source train samples `0,11,15,22,27`. The gate passes with mean/min train polygon IoU `0.996028` / `0.985401`, presence/type accuracy `1.000000`, and vertex MAE `5.359486e-06`; both multi-component samples reach IoU `1.000000`. No train30 run was executed, and no polygon inverse candidate is promoted.

S274-S278 runs the first polygon inverse train30 / val10 / test10 quick probe. Train mean/min polygon IoU is `0.731445` / `0.518519`, below the train-fit gate, although train presence/type accuracy is `1.000000` / `1.000000`. Val/test IoU is observation-only and remains weak at `0.033122` / `0.089484`. The next step is not multi-seed validation or candidate promotion; it is a targeted polygon vertex train-fit repair plan.

S279-S283 repairs the polygon inverse train30 fit gate. S280 confirms the S275 failure was broad vertex/edge precision under hard-raster sensitivity, not a polygon target or ordering failure. The minimal `longer_train30` repair uses the same S275 configuration with `steps=20000` and passes train fit: train mean/min IoU is `0.935101` / `0.802920`, presence/type accuracy is `1.000000` / `1.000000`, and train vertex MAE is `5.560893e-05`. Val/test remain observation-only and weak; this stage does not promote a polygon inverse candidate or replace the S185/S181 center-bin branch candidate.

S284-S288 diagnoses why polygon inverse generalization remains weak after train30 fit is repaired. The coarse split design and signal scale are not obviously broken: hard-case labels, rotated/multi-component rates, and Bz std are same-scale across train/val/test. The failure is instead held-out vertex/shape instability: val/test have `4/10` and `6/10` zero-IoU samples, vertex MAE is two orders of magnitude above train, and predictions show signed-area flips plus occasional out-of-grid vertices. The next stage should plan output-shape / vertex-parameterization repair or controlled resplit diagnostics, not multi-seed validation.

S289-S293 implements a center-anchored polygon representation to separate component localization from local shape prediction. Center/local decode preserves polygon oracle IoU at `1.000000`, and the new independent runner passes one-sample, five-sample, and train30 gates; train30 mean/min IoU is `0.989276` / `0.857143`. Held-out val/test IoU remains weak at `0.072402` / `0.084416`, but signed-area flips and out-of-grid vertices drop to `0`; the remaining bottleneck is held-out center-bin/local-shape generalization. The S185/S181 center-bin candidate and the absolute-vertex polygon runner remain unchanged.

S294-S297 diagnose that center-anchored held-out failure without new training. All `16/16` val/test zero-IoU samples have at least one center-bin error, and all `16/16` have y-bin errors; x-bin errors affect `8/16`. Matched-coverage analysis shows `19` uncovered held-out component bins, with `15/16` zero-IoU samples touching uncovered train center-bin coverage. The next unique recommendation is a matched-coverage resplit gate using the existing polygon V3 pack before adding model complexity, steps, multi-seed validation, or larger COMSOL data.

S298-S302 executes that matched-coverage resplit gate on the existing 50-sample polygon V3 pack. The resplit preserves hard-case counts at train/val/test `10/5/7/5/3`, `3/2/2/2/1`, and `3/2/2/2/1`, and makes all held-out component bins within train center-bin distance `<=1`. The unchanged center-anchored runner still fits train (`0.995598` / `0.969697` mean/min IoU) but held-out val/test stay weak (`0.037245` / `0.072368`) with zero-IoU `8/10` / `9/10`; all `17/17` matched-split zero-IoU samples still have y-bin errors. The next route should repair center-anchored y-bin localization rather than running multi-seed, adding steps, or expanding COMSOL data.

S303-S307 tests that y-bin localization repair with default-off soft-target losses. The same-run reference exactly reproduces S300. `neighbor_soft_y` partially improves val/test y-bin acc from `0.230769` / `0.083333` to `0.307692` / `0.166667` and zero-IoU from `8/10` / `9/10` to `7/10` / `8/10`, but it does not improve both val/test IoU over reference; `distance_soft_y` is worse. The repair gate fails, so the next branch-local route should target local-shape conditioning / bounded local output rather than more y-loss tuning, multi-seed validation, or more COMSOL data.

S308-S312 tests bounded local vertex output for the center-anchored polygon runner. `raw` mode preserves previous behavior; `bounded_tanh` optionally maps local head outputs through fixed-grid or train-stats bounds before local vertex loss, decode, metrics, and prediction export. The same-run reference reproduces val/test IoU `0.037245` / `0.072368`; `bounded_local_fixed_grid` reaches `0.024490` / `0.060554`, and `bounded_local_train_stats` reaches `0.029174` / `0.067532`. The gate fails without out-of-grid vertices or signed-area flips, so the next branch-local route should target local-shape conditioning rather than more bound sweeps, y-loss tuning, multi-seed validation, or new COMSOL data.

S313-S317 tests default-off local-shape conditioning for the center-anchored polygon runner. The runner now supports `--local-shape-conditioning-mode none|center_bin|center_bin_slot|center_bin_slot_type`, with `none` preserving previous behavior. The same-run reference exactly reproduces S311, but the first conditioned variant, `conditioning_center_bin`, improves train/local fit while reducing held-out val/test IoU to `0.027215` / `0.067059` and increasing val zero-IoU to `9/10`; slot and type variants are therefore skipped by stop condition. The next branch-local route should repair the coupled center-bin/local-shape mechanism, not continue simple local conditioning, y-loss, bound sweeps, multi-seed validation, or new COMSOL data.

S318-S322 diagnoses that coupled center-bin/local-shape mechanism. Offline oracle ablation shows center decode is the dominant held-out bottleneck: on the matched split reference, replacing GT center bin plus offset raises val/test IoU from `0.037245` / `0.072368` to `0.450778` / `0.438502`, while replacing only local vertices reaches only `0.058471` / `0.095985`; full GT center and local vertices recover polygon oracle IoU `1.000000`. A single default-off `soft_center_scheduled` joint repair was tested after the ablation, but it failed the gate: train mean IoU fell to `0.977544`, val collapsed to `0.000000`, and test only reached `0.090810`. The next branch-local route should design a stronger explicit center-local coupling mechanism rather than running more ad hoc variants, multi-seed validation, extra steps, or new COMSOL data.

S323-S327 tests the narrower loss-side version of that idea: default-off soft decoded-center consistency. The same-run reference reproduces S321 exactly with train IoU `0.995598` / `0.969697` and val/test IoU `0.037245` / `0.072368`. `soft_decoded_center_consistency` improves train center error but fails the gate: train IoU drops to `0.983633` / `0.857143`, val IoU collapses to `0.000000`, and test IoU falls to `0.034211`. `soft_decoded_vertex_consistency` is skipped by stop condition. The next route should be a structural component-query center/shape head or equivalent shared component representation, not more loss-weight tuning.

S328-S332 implements that independent component-query center/shape route. The new model and runner smoke tests pass, but the first 1-sample hard-raster gate does not: sample `0` reaches presence/type/x-bin/y-bin accuracy `1.000000` and decoded vertex MAE `5.918177e-06`, while polygon IoU is `0.974227` against the required `>=0.99`. The 5-sample, same-run reference, and train30 gates are skipped by stop condition. The next step should diagnose component-query one-sample raster sensitivity before scaling the route.

S333-S335 diagnoses that one-sample miss offline without new training. The S330 prediction is reproduced exactly: IoU `0.974227`, pred/target area `194` / `189`, with `5` false-positive pixels and `0` false-negative pixels. `gt_center + pred_local_vertices` and centroid alignment both recover IoU `1.000000`, while `pred_center + gt_local_vertices` stays at `0.979275`; area-only and edge-only scaling also stay at `0.979275`. The next route should be a 1-sample center / centroid precision repair before any 5-sample or train30 gate.

S336-S340 tests that 1-sample center / centroid precision repair with default-off component-query auxiliary losses. The same-run reference exactly reproduces S330: IoU `0.974226804`, pred/target area `194 / 189`, and presence/type/x-bin/y-bin accuracy `1.000000`. `decoded_center_aux_small` improves IoU to `0.984126984` but shifts the raster error to `0 / 3` FP/FN and area `186 / 189`; `polygon_centroid_aux_small` is worse at IoU `0.963917526`. The gate remains blocked, so 5-sample and train30 are still not run.

S341-S345 tests a narrower boundary precision repair on the same 1-sample gate. `center_aux_half` reaches IoU `0.989528796`, pred/target area `191 / 189`, and FP/FN `2 / 0`, but still misses the explicit `>=0.99` gate; `center_aux_half_plus_tiny_area` falls to IoU `0.979166667`. The route remains blocked before 5-sample/train30. The next step should be either a truly boundary-aware 1-sample repair or a deliberate review of whether `>=0.99` is too strict for a 189-pixel mask.
