# PINN_project

本仓库是 MFL（magnetic flux leakage，漏磁）缺陷几何反演项目。当前研究主线已经从“2D / quasi-2D mask 或 profile 小修补”切换到 **true 3D / Piao-style geometry profile**：用 COMSOL 生成 Bx / By / Bz forward 数据，围绕 RBC-style 六参数 `L/W/D/wLD/wWD/wLW` 做 3D profile 反演、表示审计和 profile-level 评价。

详细实验历史放在 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。本文件只记录当前状态、baseline、主线阶段、文档导航和运行注意事项；长期流水账以 `results/summaries/` 和实验日志为准。

## 当前 CURRENT_BASELINE

当前 baseline 以 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 为准。第 20.86 已完成 baseline transition：项目主线从旧的 2D v3_complex mask / boundary prediction 切换到 **true 3D RBC-style profile-depth reconstruction**。

- task: true 3D RBC-style defect profile/depth reconstruction
- dataset_id: `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`
- input: Bx/By/Bz `delta_b`, shape `(N, 3, 3, 201)`，Conv1D 视图 `(N, 9, 201)`
- model family: 20.77 small Conv1D encoder + MLP six-parameter head
- output: `L_m/W_m/D_m/wLD/wWD/wLW` -> RBC-style 3D profile/depth -> projected mask
- formal rerun: 20.85，selected seed `42`
- main metric: profile/depth reconstruction，而不是 2D mask Dice 或逐项 `wLD/wWD/wLW` MAE

当前 baseline 的关键 test 指标：normalized MAE `0.678014`，profile depth RMSE `0.000387737 m`，Er-like profile error `0.340544`，L/W/D MAE `1.892/2.186/0.800 mm`，projected mask IoU/Dice `0.750650/0.847727`。`wMAE=0.201076` 只作为 auxiliary diagnostic。

旧 v3_complex mask-only grid decoder + forward consistency baseline 没有删除，已在 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 中降级为 archived comparator。它仍可用于 2D mask / boundary 历史对比，但不再是当前主线 baseline。

如果 README 与 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 冲突，以 CURRENT_BASELINE 为准。

## 当前主线阶段

第 20.61-20.64 已连续验证：single-height Bz、multi-height Bz、same-direction Bx/By/Bz、multi-direction excitation 都没有让真实 COMSOL oracle residual 稳定排序 2D profile quality。因此当前正式暂停 2D profile-forward 小修，不再继续调 2D profile surrogate、refinement loss、direction weighting 或同类 perturbation pack。

第 20.66-20.72 已打通 true 3D RBC-style imported-watertight 数据链路：`RBC params -> depth map -> watertight mesh / COMSOL imported solid -> Bx/By/Bz @ sensor_z_m=0.008 -> delta_b -> NPZ/schema/registry/manifest`。这些 pack 当时仍是 generated data / pilot candidate，不提交 raw COMSOL CSV、`.mph`、NPZ 或 preview artifact。

第 20.77-20.85 已在固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上完成 training gate、benchmark candidate audit、feature / fusion / representation diagnostics，以及 20.77 candidate 的 formal rerun。20.85 稳定复现 20.77：selected seed `42`，test normalized MAE `0.678014`，`L/W/D` MAE `1.892/2.186/0.800 mm`，projected mask Dice `0.847727`，profile depth RMSE `0.000387737 m`。主要风险仍集中在 `wLD/wWD/wLW` curvature 表示，特别是 boxy / sharp profile。

第 20.86 已把 20.77 / 20.85 的 profile-depth candidate 升级为 CURRENT_BASELINE。20.81 只作为 projected-mask / visual comparator：Dice 更高 `0.866573`，但 profile RMSE 更差 `0.000445297 m`。20.83 是 profile-primary negative gate：Dice `0.868042`，但 profile RMSE `0.000409718 m` 仍不如 20.77 / 20.85，因此不替代 baseline。

dense mask / old 2D baseline 只保留为 archived comparator，不再作为当前 true 3D / geometry-forward 主线。

第 25.18 已对 surface multi-pit component-set 分支做路线重置：停止把 per-component raster ownership 当作主监督继续微调。证据链是 25.13 target-v2 的 near-empty mask collapse、25.15 label-v3 的 union-like merged collapse，以及 25.17 label-v3b 仍保持 merged rate `1.000000`。这不是停止 multi-pit 方向，而是把下一阶段主线切换为 **geometry-primary component-set**：保留 `K=3` slots 和 component geometry prediction，把 mask/depth 改为由几何 slot 派生的评价/弱监督对象。

## Piao / true 3D 路线口径

Piao-style 原路线可以概括为：

```text
three-axis MFL -> NLS physics features -> LS-SVM -> RBC six-parameter 3D profile
```

本项目当前只迁移其中可落地的部分：three-axis MFL observation、RBC six-parameter 3D profile label、geometry parameter regression、projected 2D mask QA、forward consistency。不要把当前工作写成完整复现 Piao 2019；当前 COMSOL 数据仍标记为 `exact_piao_rbc=False`、`rbc_style_approximation=True`，几何实现以 imported watertight mesh solid 路线为主。

当前 3D representation 仍输出 Piao RBC six params：`L, W, D, wLD, wWD, wLW`。但 20.82 后，profile/depth reconstruction error 是更适合的主评价；`wLD/wWD/wLW` 作为 curvature diagnostics，不再单独承担 true 3D branch 的 headline 成败判断。projected mask 是 2D footprint QA，不能替代 true 3D profile 指标。

## COMSOL Data-Domain References

COMSOL 数据域 baseline / reference 与当前 true 3D RBC `CURRENT_BASELINE` 分开记录，不互相替换：

- [COMSOL_DATA_BASELINE.md](COMSOL_DATA_BASELINE.md): single-defect COMSOL data-domain baseline。
- [COMSOL_DATA_BASELINE_V2.md](COMSOL_DATA_BASELINE_V2.md): single + `component_count=2` combined COMSOL data-domain reference。
- [COMSOL_MULTI_DEFECT_DATA_BASELINE.md](COMSOL_MULTI_DEFECT_DATA_BASELINE.md): true two-component multi_defect COMSOL reference。
- [COMSOL_THREE_COMPONENT_DATA_BASELINE.md](COMSOL_THREE_COMPONENT_DATA_BASELINE.md): true three-component multi_defect COMSOL reference。
- `COMSOL_DATA_BASELINE_V3.md` 尚未创建；combined single + cc2 + cc3 candidate 未通过 baseline acceptance。

这些 COMSOL data-domain reference 主要证明数据链路、schema 或局部 feasibility；第 20.86 的 CURRENT_BASELINE transition 只针对固定 v3_240 true 3D RBC profile-depth benchmark，不等于真实实验部署结论。

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
- surface multi-pit 分支中以 label-v2 / label-v3 / label-v3b per-component raster target 作为主监督继续训练；
- loss rebalance / label-v4 raster-target 微调作为 surface multi-pit 的主线；
- 将 `wLD/wWD/wLW` 逐项 MAE 作为 true 3D branch 唯一主评价；
- 在没有 explicit formal rerun / report package / baseline transition 记录的情况下直接把 pilot 数据或模型升级为 baseline。

这些结论的细节见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)、[PINN优化路线.md](PINN优化路线.md) 和 `results/summaries/`。

## 文档导航

- [CURRENT_BASELINE.md](CURRENT_BASELINE.md): 当前 authoritative baseline、reference baseline、关键指标和限制。
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): 实验历史、阶段结论和关键结果。
- [NEXT_STEP.md](NEXT_STEP.md): 最近下一步，不堆长期历史。
- [PINN优化路线.md](PINN优化路线.md): 路线级判断、已停止方向、阶段切换原则。
- [术语说明.md](术语说明.md): 项目术语、baseline、forward consistency、COMSOL 数据包、true 3D / Piao-style 等定义。
- [REPRODUCTION_FILE_MANIFEST.md](REPRODUCTION_FILE_MANIFEST.md): 在另一台电脑复现当前缺陷预测链时需要带走的文件清单。
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

1. 先读 [NEXT_STEP.md](NEXT_STEP.md) 确认最近任务；当前 surface multi-pit 下一步是 25.19 geometry-primary component-set 设计与标签派生计划，不训练。
2. 需要判断 baseline 时读 [CURRENT_BASELINE.md](CURRENT_BASELINE.md)。
3. 需要判断路线是否该继续时读 [PINN优化路线.md](PINN优化路线.md)。
4. 需要追溯实验历史时读 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 和相关 `results/summaries/`。

如果文档与记忆不一致，以仓库内当前 Markdown 和结果记录为准。
