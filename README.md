# PINN_project

本仓库是 MFL（magnetic flux leakage，磁通泄漏）缺陷边界形状反演项目。当前目标是从 Bz / multi-line delta_Bz 等观测信号反演 2D / quasi-2D defect mask / boundary，而不是普通图像分割，也不是单纯最小化完整 mu 场的 MSE。

详细实验历史放在 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。本文件只作为项目入口页，记录当前主线、baseline、文档导航和运行注意事项。

## 当前 CURRENT_BASELINE

当前 v3_complex 主线 baseline 以 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 为准：

- model family: mask-only grid decoder + forward consistency
- dataset: `v3_complex`
- forward surrogate: `checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`
- `lambda_forward = 0.10`
- validation-selected probability threshold = `0.80`
- threshold selection source: validation set only

该 baseline 通过 frozen mask-to-Bz forward surrogate 约束预测 mask 能解释观测 Bz。它相比上一版 mask-only grid decoder + threshold `0.90` 同时改善 IoU、Dice、area_error、center_error 和 Bz MSE。

仍需明确保留的限制：

- polygon / rotated_rect 精细边界圆斑化仍未根本解决；
- polygon area_error 存在轻微 trade-off；
- small / low-signal 与 multi-defect 仍是困难样本；
- 它不是“边界问题已完全解决”的结论。

## 历史 Reference Baselines

这些 baseline 保留为参考，不再代表当前主线：

- `v3_complex_tv_sweep_2e-6`: MSE-oriented reference，用于完整 mu field 数值误差参考。
- composite-selection: mu-threshold shape-oriented reference，用于对照完整 mu field + threshold 路线。
- mask-only grid decoder + threshold `0.90`: previous boundary baseline，用于对照第 18.4 之前的 boundary model。
- mask-only MLP boundary model: earlier boundary reference。

如果 README 与 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 冲突，以 CURRENT_BASELINE 为准。

## COMSOL Data-Domain References

COMSOL 数据域 baseline / reference 与 v3_complex `CURRENT_BASELINE` 分开记录，不互相替换：

- [COMSOL_DATA_BASELINE.md](COMSOL_DATA_BASELINE.md): single-defect COMSOL data-domain baseline。
- [COMSOL_DATA_BASELINE_V2.md](COMSOL_DATA_BASELINE_V2.md): single + `component_count=2` combined COMSOL data-domain reference。
- [COMSOL_MULTI_DEFECT_DATA_BASELINE.md](COMSOL_MULTI_DEFECT_DATA_BASELINE.md): true two-component multi_defect COMSOL reference。
- [COMSOL_THREE_COMPONENT_DATA_BASELINE.md](COMSOL_THREE_COMPONENT_DATA_BASELINE.md): true three-component multi_defect COMSOL reference。
- `COMSOL_DATA_BASELINE_V3.md` 尚未创建；combined single + cc2 + cc3 candidate 未通过 baseline acceptance。

## 当前主线阶段

项目已经进入第 20 阶段：forward 数据增强 / COMSOL multi-line forward data。

阶段目标已经从“继续在当前单条 Bz + decoder 上做小修补”切换为“提高反演问题本身的可辨识性”。当前重点是构建真实、可审计、schema 完整的 COMSOL forward 数据包，并验证 PINN_project 能否稳定完成：

```text
读取 NPZ -> dataset loader -> train-only normalization -> training gate -> validation threshold selection -> test smoke evaluation -> preview
```

最近主线状态：

- 已完成 COMSOL 8-sample small pack，验证 schema / tiny training smoke。
- 已完成 36-sample rectangular_notch pilot pack，验证 pilot training gate。
- 已完成 120-sample rectangular_notch pilot_v2 pack，验证 train-only normalization 和 pilot_v2 training gate。
- 已完成 rotated_rect / angle variation pilot_v3 数据生成与 ingest / training gate。
- 已完成 single / `component_count=2` / `component_count=3` COMSOL 数据域 reference 与 combined V3 candidate 审计；V3 candidate 未通过 baseline acceptance。
- 当前 geometry / forward-consistency experimental direction 是：显式 geometry representation + differentiable rasterization + forward residual / coarse-to-fine refinement。
- COMSOL_Multiphysics_MCP 是外部 forward generation 工程，不属于 PINN_project 主线训练仓库。

这些 COMSOL pilot 结果只证明数据链路和训练入口可用，不更新 v3_complex CURRENT_BASELINE，也不能直接作为正式泛化结论。

## 已停止方向

以下方向已经记录为收益不足、不稳定，或不适合作为下一条主线；除非有明确阶段级新证据，不再继续做小修补：

- loss / threshold / selection metric 小修补；
- adaptive threshold / threshold trick；
- SDF / boundary head；
- coordinate refinement；
- hand-crafted Bz features；
- U-Net-like decoder；
- shape-type conditional decoder；
- retrieval / latent shape prior；
- star-convex radial model；
- single rotated box / deformable quad / oracle quad supervised geometry；
- profile-band representation；
- anisotropic basis direct decoder；
- proposal refinement / mask-logit refinement；
- 继续围绕当前 grid decoder 做小 head / 小 loss / 小 threshold 修补。

这些结论的细节见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)、[PINN优化路线.md](PINN优化路线.md) 和 `results/summaries/`。

## 文档导航

- [CURRENT_BASELINE.md](CURRENT_BASELINE.md): 当前 authoritative baseline、reference baseline、关键指标和限制。
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): 实验历史、阶段结论和关键结果。
- [NEXT_STEP.md](NEXT_STEP.md): 最近下一步，不堆长期历史。
- [PINN优化路线.md](PINN优化路线.md): 路线级判断、已停止方向、阶段切换原则。
- [术语说明.md](术语说明.md): 项目术语、baseline、forward consistency、COMSOL 数据包等定义。
- [COMSOL_DATA_BASELINE_V2.md](COMSOL_DATA_BASELINE_V2.md) / [COMSOL_THREE_COMPONENT_DATA_BASELINE.md](COMSOL_THREE_COMPONENT_DATA_BASELINE.md): 当前 COMSOL data-domain reference 文档。
- [AGENTS.md](AGENTS.md): Codex 执行规范、Git / 数据安全、Markdown 文档管理规则。
- README.md: 项目入口页和当前状态摘要。

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

- 关键脚本；
- `results/metrics/` 中明确要求记录的 CSV；
- `results/summaries/` 中明确要求记录的 TXT / Markdown；
- 当前任务允许提交的 Markdown。

提交前必须查看 `git status --short`，不要使用 `git add .`。

## 当前如何继续

1. 先读 [NEXT_STEP.md](NEXT_STEP.md) 确认最近任务。
2. 需要判断 baseline 时读 [CURRENT_BASELINE.md](CURRENT_BASELINE.md)。
3. 需要判断路线是否该继续时读 [PINN优化路线.md](PINN优化路线.md)。
4. 需要追溯实验历史时读 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 和相关 `results/summaries/`。

如果文档与记忆不一致，以仓库内当前 Markdown 和结果记录为准。
