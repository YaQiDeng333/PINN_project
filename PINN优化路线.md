# PINN 优化路线

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
