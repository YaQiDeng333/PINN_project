# PINN 优化路线

## 2026-05-23 路线同步：20.62 multi-height profile oracle ordering feasibility

第 20.62 将第 20.61 的 observation-identifiability 判断进一步拆开验证：如果 single-height Bz residual 不能稳定排序 profile quality，那么先不训练 surrogate，而是直接用真实 COMSOL oracle residual 比较 multi-liftoff observation 是否更有辨识力。本轮使用 12 base / 96 profile rows，在 `sensor_z_m = [0.004, 0.008, 0.012]` 下生成 multi-height Bz；0.008m 复用第 20.61 exact observation，0.004m 和 0.012m 共 192 个 observation 使用真实 COMSOL forward。

结果显示 multi-height / multi-liftoff alone 没有解决 profile residual non-identifiability。test ordering accuracy 为：0.008m single-height `0.4909`，0.004m `0.4364`，0.012m `0.4545`，multi-height normalized `0.4545`；multi-height mismatch_rate 为 `0.5455`，residual-error correlation 为 `-0.5920`。它没有超过 20.61 single-height oracle test reference `0.5030`，也没有达到 `>0.65` 或 `+0.10` improvement gate。

路线判断因此更新为：当前问题不是 profile-compatible surrogate 训练不足，也不只是 profile perturbation data 规模不足，而是单 Bz、少量 scan line、仅改变 lift-off 的 observation 对 profile boundary quality 的排序信息仍不够。下一步若继续 forward-guided profile route，应优先转向 **multi-axis / multi-direction observation** 或更丰富 scan pattern / component，而不是训练 multi-height profile surrogate 或继续扩大同类 lift-off pack。第 20.62 仍是 POC，不更新任何 baseline。

## 2026-05-23 路线同步：20.61 expanded profile perturbation forward calibration

第 20.61 将第 20.60 的 profile-native perturbation 数据覆盖从 12 base / 96 rows 扩大到 36 base / 288 rows，其中 252 行是真实 profile polygon COMSOL forward，36 行 `true_reference` 复用 pilot_v9 原始数组作为 residual anchor。该实验仍是 forward surrogate calibration POC：不做 profile refinement，不训练 inverse model，不更新 baseline。

结果显示扩大数据确实缓解了 20.60 的极端 test collapse：selected `EPPF1_profile_station_mlp` 的 test surrogate ordering 从 20.60 的 `0.2143` 提升到 `0.5569`，test mismatch_rate 从 `0.7857` 降到 `0.4431`，waveform val/test correlation 也保持在 `0.9435 / 0.9299`。但是 strict gate 未通过，validation ordering 只有 `0.5361`，test ordering 也未达到 `>0.65`；更关键的是 COMSOL oracle residual ordering 在 train/val/test 只有 `0.4471 / 0.5120 / 0.5030`，说明当前观测配置下真实 residual 本身无法稳定排序 profile quality。

路线判断因此更新为：profile-compatible surrogate 的 waveform fit 不是主要 blocker，当前瓶颈转为 **profile residual objective / observation identifiability**。在 3 条 scan line、单 Bz、constant-depth top-view profile polygon 设置下，继续扩同类 profile perturbation data 或小调 surrogate architecture 的收益有限；下一步如果继续 forward-guided route，应优先转向 richer observations（multi-height / multi-axis / more scan lines）或专门做 non-identifiability audit。第 20.61 不支持直接回到 profile-forward refinement retry，也不改变现有 baseline。

## 2026-05-22 路线同步：20.60 profile perturbation forward calibration

第 20.60 将第 20.59 的结论进一步落地：如果要让 forward residual 支撑 profile-basis refinement，校准数据必须围绕 profile representation 本身，而不是继续复用 rect/rot geometry perturbation。为此本轮设计了 24 base / 192 rows 的 profile perturbation plan，并在 COMSOL 侧按 minimum partial 生成 12 base / 96 rows forward pack，其中 84 行是真实 profile polygon COMSOL forward，12 行 `true_reference` 复用原始 pilot_v9 作为 residual anchor。

该实验确认了两点。第一，profile polygon generation 和 forward-pack schema 本身可行：split/type/variant coverage 达到 minimum partial，`delta_bz = bz_defect - bz_no_defect` 校验通过，且没有把 profile perturbation 退回 single rotated box。第二，当前数据规模仍不足以支撑 profile-forward refinement：`PPF1_profile_station_mlp` 的 waveform val/test NRMSE 为 `0.4396 / 0.3758`，但 residual ordering test collapse 到 `0.2143`，mismatch_rate 达到 `0.7857`；同时 COMSOL oracle residual test ordering 也只有 `0.5357`，说明在当前 2 个 test base 的 partial pack 上，真实 residual 对 profile quality 的排序信号也不稳定。

路线判断因此更新为：profile-compatible surrogate 方向没有被实现错误否定，但当前 partial pack 不能支撑 profile-forward refinement retry。下一步若继续 forward-guided profile route，必须先扩大 profile perturbation data，特别是增加 val/test base 覆盖并重新验证 oracle ordering；若扩展后 oracle residual 仍弱，则主要瓶颈可能是观测 non-identifiability，需要 richer observations / multi-axis / multi-height，而不是继续调 surrogate architecture 或 refinement loss。第 20.60 仍是 POC，不更新 baseline。

## 2026-05-22 路线同步：20.58 mask/profile basis refinement

第 20.58 将第 20.57 否定的 single rect/rot low-dimensional refinement，替换为从 dense/coarse initializer 提取的 K=8 mask/profile basis 表示。该路线的目标是避免单个 rotated rectangle 参数空间过窄，同时不回到完全自由的 dense mask decoder。

结果显示 profile basis 有边际价值：profile extraction 基本保留 dense proposal，no-forward profile refinement 在 test 上达到 `0.6697 / 0.8002 / 0.2196`，好于第 20.57 calibrated rect/rot refinement `0.6492 / 0.7829 / 0.2417`，也接近第 20.54 strong dense initializer。它说明 profile/basis 表示能作为更柔性的低维形状空间，但尚未稳定超过第 20.54 extracted rotated-box proposal `0.6726 / 0.8017 / 0.1945`。

更关键的是 forward profile refinement 的 validation sweep 选择了 `lambda_forward=0.0`，说明当前 S1 perturbation-calibrated surrogate 通过 lossy profile-to-rect summary 接入后，不能可靠驱动 profile-space optimization。路线判断因此更新为：当前瓶颈不是再换一个更复杂的 shape basis，而是需要 profile-compatible forward surrogate 或 richer observations。继续在 current surrogate + current profile space 上调 steps、lr、lambda_forward 意义有限；下一步若继续 geometry/refinement route，应优先改进 forward surrogate 的输入表示和 residual calibration，而不是继续增加 direct geometry head 或单 box refinement 变体。

## 2026-05-22 路线同步：20.57 calibrated refinement retry

第 20.57 验证了一个关键负面结果：第 20.56 的 perturbation-calibrated `S1_perturb_geom_mlp` 虽然能复现较好的 pairwise residual ordering（val/test `0.7321 / 0.8036`），但把它放进连续低维 Priewald-style refinement 后，mask / geometry 指标没有同步改善。以第 20.54 的 strong dense/extracted proposal 为初值，test geometry-raster IoU/Dice 从 `0.6726 / 0.8017` 下降到 `0.6492 / 0.7829`；forward residual 继续下降，但 mismatch_rate 达到 `0.6212`，residual reduction 与 IoU/Dice delta 呈负相关。

路线判断因此更新为：当前瓶颈不只是 surrogate waveform fit，也不只是 pairwise ordering，而是 residual objective 在连续 rect/rot parameter space 中的可优化性。继续对 steps、lr、prior、surrogate loss 做小调意义不大；direct neural geometry head 也已经在 20.48-20.51 收口。下一步若继续 geometry-aware route，应优先转向 **mask/profile basis refinement** 或更高维形状表示；如果未来回到 Priewald-style refinement，需要先扩大 perturbation pack 或引入 richer observations，再重新证明 residual landscape 能提供稳定梯度。

## 2026-05-22 路线同步：20.56 perturbation forward calibration

第 20.56 把 20.55 的 forward surrogate mismatch 问题拆成两个问题验证：真实 COMSOL residual 是否能排序局部几何质量，以及 surrogate 是否能学到这种排序。结果显示，在 rect/rot local perturbation partial pack（96 行，84 行真实 COMSOL forward）上，COMSOL oracle residual 的 val/test ordering accuracy 为 `0.6607 / 0.8393`，选中的 `S1_perturb_geom_mlp` 的 val/test ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 相比 20.55 明显降低。

这说明当前 Priewald-style 路线的瓶颈不是“forward residual 完全无信息”，而是原先 surrogate 缺少围绕同一几何样本的局部扰动校准数据。后续如果继续 refinement，应使用 perturbation-calibrated surrogate 做受控 retry，并继续记录 residual-ordering、mask/geometry improvement 和 surrogate mismatch。由于当前 pack 仍是 96/192 partial pack，且 test residual-error correlation 仍为负，不能把 20.56 写成 baseline，也不能直接扩大为正式方法结论。

## 当前路线同步：第 20 阶段 forward data / COMSOL pilot

当前 `CURRENT_BASELINE` 仍是 v3_complex mask-only grid decoder + forward consistency `lambda_forward=0.10` + validation-selected threshold `0.80`。第 18.x / 19.x 已经说明继续做 decoder、loss、threshold、geometry、basis 或 refinement 小修补收益不足。

第 20 阶段的路线目标是提高反演问题本身的可辨识性：用 COMSOL / physics-forward 数据构建 multi-line `delta_Bz` -> 2D / quasi-2D mask 的可审计训练包。当前已经完成 rectangular_notch small / pilot / pilot_v2 数据链路，以及 rotated_rect / angle variation pilot_v3 数据链路。后续优先级是合并 `rectangular_notch` + `rotated_rect`，再扩展样本数和 defect_type 多样性；不要回到当前 grid decoder 的小 head / 小 loss / 小 threshold 调参。

本文件记录当前主线判断，不再按早期实验流水账追加。历史细节以 `EXPERIMENT_LOG.md`、`CURRENT_BASELINE.md` 和 `results/summaries/` 为准；本文件只保留路线层面的结论、停止条件和下一阶段原则。

## 下一阶段：forward model / 多观测数据 / COMSOL feasibility

第 18.x / 19.x 后，内部结构和 test-time refinement 路线已经基本到达边界。当前 `CURRENT_BASELINE` 仍保留为 mask-only grid decoder + forward consistency `lambda_forward=0.10` + validation-selected threshold `0.80`，它是现有 v3_complex 单通道 Bz 数据上的最强 boundary-oriented baseline。

下一阶段目标不再是继续调 decoder、loss、threshold、basis、geometry、proposal refinement 或 post-processing，而是提高反演问题本身的可辨识性。核心判断是：现有单条 / 单通道 Bz 对 polygon / rotated_rect 的直边、角点、rotation 和 multi-defect 组件约束不足，因此继续在当前输入上做小修补很难根本解决圆斑化。

推荐优先路线是 `comsol_single_defect_multiline_forward_pack_v1`：

* 先做小规模、可审计的 COMSOL / physics-forward single-defect 数据包；
* 优先覆盖 polygon / rotated_rect 或可实现的等价形状族；
* 输入优先使用 multi-line `delta_Bz`，仍输出 2D / quasi-2D defect mask；
* 先验证 Mask/Geometry -> Bz forward surrogate 是否可靠，再进入 inverse boundary model；
* 不直接上完整 3D，不直接替换当前 v3_complex `CURRENT_BASELINE`。

候选方向优先级：

1. COMSOL-generated single-defect focused dataset，最好包含 multi-line `delta_Bz`；
2. 基于该数据包训练更可靠的 forward surrogate，作为 inverse model 前置 gate；
3. three-axis MFL 作为后续扩展；
4. 单独 multi-liftoff 只作为低优先级补充，因为它更偏深度 / 尺度约束，不一定解决边界角点；
5. 不再围绕当前 grid decoder 做小 head / 小 loss / 小 threshold / 小 refinement。

该阶段的接受条件不是某个局部指标波动，而是更可靠 forward model 或更丰富观测能否稳定改善边界可辨识性：预测 mask 是否更能解释 Bz，同时 IoU / Dice / area_error / small-low-signal / polygon-rotated_rect 视觉质量是否优于当前 baseline。

## 当前主线状态

当前 `CURRENT_BASELINE` 已从早期的 μ-field threshold / composite-selection 路线，以及上一版 mask-only grid decoder boundary baseline，更新为：

* 模型族：mask-only grid decoder + forward consistency
* 数据集：`v3_complex`
* forward surrogate：`checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`
* `lambda_forward = 0.10`
* 选定 probability threshold：`0.80`
* threshold 选择来源：仅 validation set

本项目目标是缺陷边界形状反演，因此 baseline 选择优先看 IoU、Dice、area_error、`pred_area=0`、small / low-signal 表现，而不是只看 μ-field 的 MSE / MAE。

保留参考线：

* `v3_complex_tv_sweep_2e-6` 保留为 MSE-oriented reference，不再是主线 `CURRENT_BASELINE`。
* composite-selection 保留为 μ-threshold shape-oriented reference，用于对照完整 μ-field threshold 方案。
* 前一版 mask-only MLP boundary model 保留为 boundary reference。

当前 forward consistency baseline 的核心结论是：在 mask-only grid decoder 基础上加入 frozen mask-to-Bz surrogate consistency 后，整体 IoU / Dice / area_error / center_error 与 Bz MSE 同时优于上一版 grid decoder baseline，说明预测 mask 能更好解释观测到的 Bz 信号。但它仍没有根本解决边界精细形状，polygon area_error 轻微恶化，polygon / rotated_rect 仍可能被预测成偏圆的平滑斑块，multi-defect 和 small / low-signal 样本仍是难点。

## 已停止路线

以下方向已经完成阶段性验证或被明确停止，不再继续做小修补：

* v4 / calibrated_mu / enhanced decoder
* threshold trick / adaptive threshold
* aux mask head 作为最终输出
* aux mask loss / shape-aware loss 迁移到 v3
* defect-weighted MSE / threshold-margin loss
* small oversampling
* CNN1D BzEncoder
* hand-crafted Bz features
* multi-liftoff
* geometry auxiliary supervision
* warm-start / curriculum
* selection metric 细调
* ensemble
* SDF / boundary loss
* coordinate refinement
* shape-prior latent
* exemplar retrieval
* star-convex radial model
* U-Net-like decoder
* shape-type conditional decoder
* 继续围绕当前 grid decoder 做小 head / 小 loss / 小 threshold 修补

停止这些方向的共同原因是：它们要么只带来局部指标波动，要么改善 IoU / Dice 的同时恶化 area_error / small / low-signal，要么视觉上仍然没有解决 polygon / rotated_rect 的圆斑化。

## Mask-Only 路线结论

mask-only boundary model 是第一次真正比 μ-threshold 路线更贴近项目目标的方向。它把任务从“先预测 μ，再用 `mu < 500` 间接取 mask”改成“直接从 Bz 预测 defect probability / mask”，因此更符合缺陷边界形状反演。

第 15.2 的 mask-only MLP + validation-selected threshold=0.90 已经明显优于 composite-selection。第 15.4 的 mask-only grid decoder 进一步提升，并成为当时的 boundary-oriented `CURRENT_BASELINE`。第 18.4 之后，mask-only grid decoder + forward consistency `lambda_forward=0.10` 进一步成为新的 `CURRENT_BASELINE`。

当前问题已经不是“完全找不到缺陷”，而是：

* polygon / rotated_rect 的直边、角点、旋转结构仍被圆斑化；
* multi-defect 有时漏掉组件；
* small / low-signal 样本仍不稳定；
* 继续在同一个 grid decoder 上加小 loss、小 head、小 threshold 或小 feature，收益不足以支持新主线。

## 外部研究与路线判断

Deep Research / NotebookLM 相关整理给出的共同判断是：下一阶段不应继续把问题当作普通 segmentation 小修补，而应转向更接近 MFL 反演本质的路线：

* geometry-aware inversion
* differentiable rasterization
* forward consistency
* hybrid forward-model + neural network inversion
* geometry / shape-parameter reconstruction

这些方向的共同点是：预测结果不只要像真实 mask，还要能解释观测到的 Bz 信号。也就是说，后续主线应从“mask 分割模型调参”转向“可解释 Bz 的边界 / 几何反演”。

## 当前最有价值的正方向

第 18.2 已验证 `mask -> Bz` forward surrogate 具有可用性：

* test R2 = 0.8520
* test correlation = 0.9231

这说明用预测 mask 解释 Bz 是可行的。随后 forward consistency 在第 18.2 / 18.3 / 18.4 中出现明确正信号：

* IoU / Dice 提升；
* center_error 改善；
* Bz residual / Bz MSE 明显下降；
* `lambda_forward=0.10` 在 bounded bracket check 和 3 seed validation 中表现最好；
* `pred_area=0` 没有明显恶化；
* small / low-signal 的 IoU / Dice 有改善，但 area_error 仍需谨慎看待；
* polygon 有小幅 IoU / Dice 改善但 area_error 仍是风险，rotated_rect 和 multi_defect 的改善更明确。

因此，`lambda_forward=0.10` forward consistency 已从 candidate 提升为当前 `CURRENT_BASELINE`。它是目前最有价值的正方向，但它的意义是把主线推进到 physics-consistent / hybrid inversion，而不是继续围绕同一 grid decoder 做 loss、head、threshold 或 feature 小修补。

## 下一阶段原则

后续不再做局部小修补。新实验必须围绕以下问题判断：

* 预测 mask 是否更贴合真实边界；
* 预测 mask 是否更能解释观测 Bz；
* IoU / Dice / area_error / `pred_area=0` 是否优于当前 mask-only grid decoder baseline；
* small / low-signal 是否不变差，最好改善；
* polygon / rotated_rect 的圆斑化是否真的减轻，而不是只出现局部指标波动。

forward consistency 已通过 review 和 baseline 决策，后续应进入 physics-consistent / hybrid inversion 主线。

如果后续发现 forward consistency 仍无法解决 polygon / rotated_rect 精细边界，则下一步应转向更严格的 geometry parameterization + forward consistency，而不是继续调 decoder、threshold、loss、head 或手工 Bz feature。

## 第 20.42-20.55 方法路线阶段判断

第 20.42-20.55 的结论进一步确认：外部 deep research 报告的核心路线不是普通 segmentation，也不是继续做 dense decoder patch，而是把 MFL 缺陷边界问题视为 inverse reconstruction。当前算法主线已经从 mask-only decoder / combined baseline 评估，切换到：

```text
geometry-aware representation
-> differentiable rasterization
-> forward consistency / forward-model residual
-> low-dimensional refinement
```

几个关键判断如下：

* combined COMSOL_DATA_BASELINE_V3 lightweight decoder 失败主要来自 `component_count=2` 与其他拓扑任务的冲突，不应通过继续加宽普通 decoder 解决。
* topology-gated decoder v1/v2 只是 weak topology-aware decoder patch。它没有显式 geometry 参数、没有 differentiable rasterization、没有 predicted geometry 到 Bz 的 forward residual，因此不属于外部报告的核心 geometry-aware / forward-consistent 方法。
* Piao 2019 当前只适合作为弱适配探索：本项目只有 multi-line Bz / quasi-2D geometry，而论文核心是三轴 MFL、RBC 3D profile、NLS 物理特征和 LS-SVM。20.47-revised 的 Bz-only NLS-style features + SVR/KRR/Ridge 没有通过 acceptance，后续若进入 3D / 三轴数据阶段再考虑更深入的 Piao-style 方法。
* 20.48 证明 differentiable rotated-rectangle rasterizer 与 geometry labels 可用，说明 geometry-aware route 本身有可行性；但 20.49 / 20.50 说明继续修 direct neural geometry head 难以解决 type / angle 学习不足。
* 20.51 的 feature-assisted geometry head + lightweight forward consistency 只带来边际 mask / angle 改善，type confusion 仍未解决。因此 direct Bz -> geometry head 不应继续小修补。
* 20.52 证明 Priewald-style low-dimensional refinement 有正信号：frozen forward surrogate residual 能显著降低 forward NRMSE，并带来小幅 geometry-raster mask 改善，但 geometry-head initializer 偏弱。
* 20.53 的 dense/coarse initializer + refinement 进一步说明，refinement 上限主要受 proposal 质量限制。当前 binary dense mask + PCA rotated bbox extraction 没有超过 20.51 geometry-head proposal，type / angle 初始化仍弱，因此不能作为 candidate 或 baseline。
* 20.54 用 strong dense initializer + improved proposal extraction 显著修复了 proposal 质量：geometry-raster test IoU/Dice 达到 `0.6726 / 0.8017`。但从该强 proposal 做 Priewald-style refinement 后，forward NRMSE 下降而 mask IoU/Dice 回落到 `0.6646 / 0.7958`，说明当前主要 bottleneck 已从 initializer/proposal extraction 转为 forward surrogate mismatch / non-identifiability。
* 20.55 进一步确认 bottleneck 不是简单 waveform fit，而是 residual objective calibration：S1/S2/S3 三个 calibrated surrogate candidate 均未让 residual 与 geometry/mask error 建立非平凡正相关，Stage C refinement 因 gate 未过被跳过。这个阴性结果说明继续调当前 surrogate loss 或 refinement objective 意义有限。

Priewald 2013 对当前阶段更重要的启发不是复现完整 FEM、解析 Jacobian 或 Gauss-Newton 工程，而是 forward-model-based inversion / refinement：用 predicted geometry 通过 forward surrogate 生成 MFL，再用 observed MFL residual 约束低维几何参数。当前应停止 direct neural geometry head 小修补；20.55 之后若继续 Priewald-style refinement，前置条件应是 synthetic perturbation forward data 或等价局部扰动数据，让 surrogate 学到 geometry perturbation 与 signal residual 的局部排序关系。若无法补足这个 calibration 证据，则转向 mask/profile basis refinement，避免被低保真 residual objective 牵引到错误几何。
## 第 20.59 方法路线判断：profile-compatible forward surrogate

第 20.59 将第 20.58 的结论进一步拆开验证：profile basis 本身仍有价值，但 forward consistency 必须换成 profile-compatible forward surrogate，不能把 K=8 profile stations 再压缩成 single rotated-box summary。外部文献路线，尤其 Priewald-style forward-model-based inversion，支持这种判断：关键不是复现完整 FEM Jacobian，而是让 forward residual 对待优化的 shape/profile representation 有一致、可校准的响应。

本轮使用已有 pilot_v9 original samples 和 20.56 perturbation pack 构建 profile-forward dataset，没有运行 COMSOL，也没有生成新数据。`PFS3_profile_station_sequence` 的 waveform fit 可接受（val/test NRMSE `0.3841 / 0.3995`），说明 profile-native 表示可被 forward surrogate 消化；但 validation residual ordering accuracy 只有 `0.6607`，mismatch_rate 为 `0.3393`，未达到 refinement gate。因此第 20.59 不执行 profile-forward refinement retry。

路线判断：profile-compatible surrogate 相比旧 rect-like bridge 有边际价值，但当前 perturbation coverage 太小，不足以支撑连续优化。下一步若继续 forward-guided profile refinement，应先扩展 profile perturbation data；否则保留 20.58 的 no-forward profile basis 作为更稳的 representation 证据，并暂停对当前 forward residual objective 的小调。该阶段仍是 POC，不更新任何 baseline。
