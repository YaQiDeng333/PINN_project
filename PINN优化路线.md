# PINN 优化路线

本文件记录当前主线判断，不再按早期实验流水账追加。历史细节以 `EXPERIMENT_LOG.md`、`CURRENT_BASELINE.md` 和 `results/summaries/` 为准；本文件只保留路线层面的结论、停止条件和下一阶段原则。

## 当前主线状态

当前 `CURRENT_BASELINE` 已从早期的 μ-field threshold / composite-selection 路线切换为：

* 模型族：mask-only grid decoder boundary model
* 数据集：`v3_complex`
* checkpoint family：
  * `checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed42.pt`
  * `checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed123.pt`
  * `checkpoints/mask_boundary_grid_candidate/best_mask_boundary_grid_seed2026.pt`
* 选定 probability threshold：`0.90`
* threshold 选择来源：仅 validation set

本项目目标是缺陷边界形状反演，因此 baseline 选择优先看 IoU、Dice、area_error、`pred_area=0`、small / low-signal 表现，而不是只看 μ-field 的 MSE / MAE。

保留参考线：

* `v3_complex_tv_sweep_2e-6` 保留为 MSE-oriented reference，不再是主线 `CURRENT_BASELINE`。
* composite-selection 保留为 μ-threshold shape-oriented reference，用于对照完整 μ-field threshold 方案。
* 前一版 mask-only MLP boundary model 保留为 boundary reference。

当前 mask-only grid decoder 的核心结论是：它能比 μ-threshold 路线更直接地预测缺陷 mask，整体 IoU / Dice / area_error / `pred_area=0` 和 small / low-signal 表现更符合边界反演目标。但它没有解决边界精细形状，polygon / rotated_rect 仍常被预测成偏圆的平滑斑块，multi-defect 和 small / low-signal 样本仍是难点。

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

第 15.2 的 mask-only MLP + validation-selected threshold=0.90 已经明显优于 composite-selection。第 15.4 的 mask-only grid decoder 进一步提升，并成为当前 boundary-oriented `CURRENT_BASELINE`。

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

因此，`lambda_forward=0.10` forward-consistency candidate 是当前最值得重点关注的下一条候选主线。但在 `CURRENT_BASELINE.md` 正式更新前，它仍是 candidate / pending review，不能擅自替换当前 mask-only grid decoder baseline。

## 下一阶段原则

后续不再做局部小修补。新实验必须围绕以下问题判断：

* 预测 mask 是否更贴合真实边界；
* 预测 mask 是否更能解释观测 Bz；
* IoU / Dice / area_error / `pred_area=0` 是否优于当前 mask-only grid decoder baseline；
* small / low-signal 是否不变差，最好改善；
* polygon / rotated_rect 的圆斑化是否真的减轻，而不是只出现局部指标波动。

如果 forward consistency 通过 review 和 baseline 决策，则进入 physics-consistent / hybrid inversion 主线。

如果 forward consistency 也失败，则下一步应转向更严格的 geometry parameterization + forward consistency，而不是继续调 decoder、threshold、loss、head 或手工 Bz feature。
