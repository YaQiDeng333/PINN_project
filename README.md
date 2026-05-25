# PINN_project

本仓库是 MFL（magnetic flux leakage，漏磁）缺陷几何反演项目。当前研究主线已经从“2D / quasi-2D mask 或 profile 小修补”切换到 **true 3D / Piao-style geometry profile**：用 COMSOL 生成 Bx / By / Bz forward 数据，围绕 RBC-style 六参数 `L/W/D/wLD/wWD/wLW` 做 3D profile 反演、表示审计和 profile-level 评价。

详细实验历史放在 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。本文件只记录当前状态、baseline、主线阶段、文档导航和运行注意事项；长期流水账以 `results/summaries/` 和实验日志为准。

## 当前 CURRENT_BASELINE

当前 v3_complex 主线 baseline 仍以 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 为准：

- model family: mask-only grid decoder + forward consistency
- dataset: `v3_complex`
- forward surrogate: `checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`
- `lambda_forward = 0.10`
- validation-selected probability threshold = `0.80`
- threshold selection source: validation set only

这个 baseline 通过 frozen mask-to-Bz forward surrogate 约束预测 mask 能解释观测 Bz。它仍是 v3_complex 上的 authoritative baseline，但没有解决 polygon / rotated_rect 细边界圆斑化、small / low-signal 不稳定、multi-defect 组件遗漏等问题。

如果 README 与 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 冲突，以 CURRENT_BASELINE 为准。

## 当前主线阶段

第 20.61-20.64 已连续验证：single-height Bz、multi-height Bz、same-direction Bx/By/Bz、multi-direction excitation 都没有让真实 COMSOL oracle residual 稳定排序 2D profile quality。因此当前正式暂停 2D profile-forward 小修，不再继续调 2D profile surrogate、refinement loss、direction weighting 或同类 perturbation pack。

第 20.66-20.72 已打通 true 3D RBC-style imported-watertight 数据链路：`RBC params -> depth map -> watertight mesh / COMSOL imported solid -> Bx/By/Bz @ sensor_z_m=0.008 -> delta_b -> NPZ/schema/registry/manifest`。这些 pack 是 generated data / pilot candidate，不是 v3_complex `CURRENT_BASELINE`，也不提交 raw COMSOL CSV、`.mph`、NPZ 或 preview artifact。

第 20.77-20.82 已在固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上完成 training gate、benchmark candidate audit、feature / fusion / representation diagnostics。v3_240 是 `train_ready_candidate=True`、`baseline_ready=False`；20.77 neural selected seed `42` 的 test normalized MAE 为 `0.678014`，`L/W/D` MAE 为 `1.892/2.186/0.800 mm`，projected mask Dice 为 `0.847727`，profile depth RMSE 为 `0.000387737 m`。主要风险集中在 `wLD/wWD/wLW` curvature 表示，特别是 boxy / sharp profile。

第 20.82 的当前口径是：true 3D / Piao-style branch 不再把 `wLD/wWD/wLW` 逐项 MAE 当作 headline metric；它们保留为 auxiliary diagnostics。主评价应转向六参数生成的 3D profile 是否准确，例如 `profile_depth_rmse_m` 或 Er-like depth/profile reconstruction error。下一步唯一建议是 **20.83 `R1_six_params_profile_primary_loss`**：仍输出六参数，但把 loss / validation 主目标改为 profile-level reconstruction；不需要新 COMSOL 数据，不扩到 480，不做 baseline replacement。

dense mask baseline 只保留为 comparator，不再作为当前 true 3D / geometry-forward 主线。

## Piao / true 3D 路线口径

Piao-style 原路线可以概括为：

```text
three-axis MFL -> NLS physics features -> LS-SVM -> RBC six-parameter 3D profile
```

本项目当前只迁移其中可落地的部分：three-axis MFL observation、RBC six-parameter 3D profile label、geometry parameter regression、projected 2D mask QA、forward consistency。不要把当前工作写成完整复现 Piao 2019；当前 COMSOL 数据仍标记为 `exact_piao_rbc=False`、`rbc_style_approximation=True`，几何实现以 imported watertight mesh solid 路线为主。

当前 3D representation 仍输出 Piao RBC six params：`L, W, D, wLD, wWD, wLW`。但 20.82 后，profile/depth reconstruction error 是更适合的主评价；`wLD/wWD/wLW` 作为 curvature diagnostics，不再单独承担 true 3D branch 的 headline 成败判断。projected mask 是 2D footprint QA，不能替代 true 3D profile 指标。

## COMSOL Data-Domain References

COMSOL 数据域 baseline / reference 与 v3_complex `CURRENT_BASELINE` 分开记录，不互相替换：

- [COMSOL_DATA_BASELINE.md](COMSOL_DATA_BASELINE.md): single-defect COMSOL data-domain baseline。
- [COMSOL_DATA_BASELINE_V2.md](COMSOL_DATA_BASELINE_V2.md): single + `component_count=2` combined COMSOL data-domain reference。
- [COMSOL_MULTI_DEFECT_DATA_BASELINE.md](COMSOL_MULTI_DEFECT_DATA_BASELINE.md): true two-component multi_defect COMSOL reference。
- [COMSOL_THREE_COMPONENT_DATA_BASELINE.md](COMSOL_THREE_COMPONENT_DATA_BASELINE.md): true three-component multi_defect COMSOL reference。
- `COMSOL_DATA_BASELINE_V3.md` 尚未创建；combined single + cc2 + cc3 candidate 未通过 baseline acceptance。

这些 COMSOL pilot 结果只证明数据链路、schema 或局部 feasibility，不更新 v3_complex CURRENT_BASELINE，也不能直接作为正式泛化结论。

## 已停止方向

以下方向已记录为收益不足、不稳定，或不适合作为下一条主线；除非有明确阶段级新证据，不再继续做小修补：

- 当前 grid decoder 的 loss / threshold / selection metric 小修补；
- adaptive threshold / threshold trick；
- SDF / boundary head / coordinate refinement；
- hand-crafted Bz features；
- U-Net-like decoder / shape-type conditional decoder；
- retrieval / latent shape prior / star-convex radial model；
- single rotated box / deformable quad / oracle quad supervised geometry；
- profile-band、anisotropic basis、mask-logit refinement；
- 2D profile-forward surrogate / refinement 的继续微调；
- multi-height Bz、same-direction multi-axis、multi-direction excitation 的同类小扩展。
- 将 `wLD/wWD/wLW` 逐项 MAE 作为 true 3D branch 唯一主评价；
- 在没有 profile-level gate 的情况下直接把 v3_240 升级为 baseline 或扩到 480。

这些结论的细节见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)、[PINN优化路线.md](PINN优化路线.md) 和 `results/summaries/`。

## 文档导航

- [CURRENT_BASELINE.md](CURRENT_BASELINE.md): 当前 authoritative baseline、reference baseline、关键指标和限制。
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): 实验历史、阶段结论和关键结果。
- [NEXT_STEP.md](NEXT_STEP.md): 最近下一步，不堆长期历史。
- [PINN优化路线.md](PINN优化路线.md): 路线级判断、已停止方向、阶段切换原则。
- [术语说明.md](术语说明.md): 项目术语、baseline、forward consistency、COMSOL 数据包、true 3D / Piao-style 等定义。
- COMSOL baseline docs: COMSOL data-domain reference 文档，不替代 v3_complex baseline。
- [AGENTS.md](AGENTS.md): Codex 执行规范、Git / 数据安全、Markdown 文档管理规则。

## 数据与 Git 注意事项

不要提交：

- `checkpoints/`
- `data/`
- `.npz`
- COMSOL `.mph`
- raw COMSOL CSV / generated raw data
- `results/previews/`
- `*.png` / `*.jpg`
- `__pycache__/`
- `notes/`

可以提交：

- 当前任务明确允许的脚本；
- `results/metrics/` 中明确要求记录的 CSV；
- `results/summaries/` 中明确要求记录的 TXT / Markdown；
- 当前任务允许提交的 Markdown。

提交前必须检查 `git status --short`，不要使用 `git add .`。

## 当前如何继续

1. 先读 [NEXT_STEP.md](NEXT_STEP.md) 确认最近任务；当前下一步是 20.83 `R1_six_params_profile_primary_loss`。
2. 需要判断 baseline 时读 [CURRENT_BASELINE.md](CURRENT_BASELINE.md)。
3. 需要判断路线是否该继续时读 [PINN优化路线.md](PINN优化路线.md)。
4. 需要追溯实验历史时读 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 和相关 `results/summaries/`。

如果文档与记忆不一致，以仓库内当前 Markdown 和结果记录为准。
