# AGENTS.md

## 项目执行规则

本 worktree 对应 `feature/dual-network-variational` 支线，用于探索主线之外的 dual-network / weak-form 反演方案。支线结论不能直接写成主线 baseline 替代结果，也不要把半监督或 diagnostic upper bound 结论表述为纯无监督成功。

每次任务开始前，优先读取当前任务直接相关的项目文档：

1. `README.md`
2. `DUAL_NETWORK_EXPERIMENT_LOG.md`，如果存在
3. `DUAL_NETWORK_STAGE_SUMMARY.md`，如果存在
4. `DUAL_NETWORK_TERMS.md`，如果存在
5. `DUAL_NETWORK_RESULTS_REPORT.md`，如果存在
6. `DUAL_NETWORK_ARTIFACT_INDEX.md`，如果存在
7. 与当前任务相关的 `experiments/dual_network/*/summary.md`
8. COMSOL 数据请求、转换、接入或 summary 类文档，如果当前任务涉及 COMSOL 数据

如果文档与记忆或口头上下文不一致，以仓库内最新文档、实验记录和实际结果文件为准。

## Claude Code review 规则

- 每次执行用户指令时，Codex 需要自行判断是否有必要调用 Claude Code 做 review。
- 不要把 Claude Code review 当成每次任务的固定步骤，也不要过于频繁调用。只有在你判断变更较关键、风险较高、涉及 baseline/benchmark/训练目标/模型结构/数据对齐，或用户明确要求 review 时，才调用 Claude Code。
- 如果判断需要 review，Codex 应自行在终端中调用 Claude Code 完成 review。
- Codex 需要根据 review 结果自行完成必要修复、补充说明或确认无需修改。
- 最终输出中只需要简要说明是否调用了 Claude Code review，以及结论或处理结果。
- 不要在输出结尾询问用户是否需要 review，也不要让用户另行手动发起 review。
- 重大 baseline 替换、benchmark 转换、可能影响项目方向的决策，仍需在执行前明确提醒用户并等待确认。

## Markdown 文档及时更新

- 每次完成实验、修复、baseline 判断、分支阶段性收口或重要设计变更后，必须及时更新相应 Markdown 文档。
- 优先更新与当前任务直接相关的文档，例如 `AGENTS.md`、`DUAL_NETWORK_EXPERIMENT_LOG.md`、`DUAL_NETWORK_STAGE_SUMMARY.md`、`DUAL_NETWORK_TERMS.md`、路线说明或 summary 类文档。
- 文档语言以中文为主；必要术语如 loss、baseline、IoU、Dice、area_error、defect_mask、mu_norm、Claude Code review、Codex 可以保留英文。
- 不要只更新代码而遗漏文档。

## Git 与数据安全

- 不要随意切换分支或破坏正在进行的实验分支。
- 不要默认运行训练、改训练脚本、改实验结果、提交大文件或 push，除非用户明确要求。
- 提交前必须先检查 `git status --short`。不要使用 `git add .`。
- 不要覆盖已有关键 checkpoint、数据包或结果文件。
- test set 只用于阶段性最终评估，不用于频繁调参。
- 涉及 `BCE mask prior`、`label-informed centers` 或 COMSOL pilot 数据时，必须明确其边界，避免夸大结论。

## 删除安全规则

禁止批量删除文件或目录。不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，一次只能删除一个明确路径的文件。如果需要批量删除文件，应停止操作，并请求用户手动删除或明确授权后再处理。

## 当前支线阶段包执行补充规则

- 当前 worktree 只处理 `feature/dual-network-variational`，不要修改 `main`，不要 push。
- 阶段包任务可以包含多个子阶段；每个子阶段完成后需要做自评，并在最终输出中报告关键结论。
- 不要频繁 checkpoint commit；除非用户要求或出现必须保留的中间状态，默认在阶段包末尾统一提交。
- 需要 Claude Code review 时，由 Codex 自行判断并调用；不要过于频繁调用，只有关键或高风险变更才调用。最终只输出 review 摘要、必须修复项和是否已修复，不输出完整 review prompt。
- 不要提交 `comsol_geometry_variation_exports/`、`comsol_geometry_variation_v2_exports/`、`comsol_pilot_exports/` 根目录。
- 不要提交 `experiments/dual_network/S35_200x100_30sample_default_validation/`、`copy-output*.md`、checkpoint、模型权重、临时文件或图片。
- COMSOL / parametric route 中的 `.npz`、CSV 或 large artifacts 只有在用户阶段规则明确允许时才提交；raw export 根目录默认不提交。
- 支线 Markdown 正文默认使用中文；文件名、参数名、指标名、命令和代码标识可保留英文。
