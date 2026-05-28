# AGENTS.md

## 项目执行规则

本项目是 MFL / 磁通漏泄缺陷边界与几何形状反演项目。当前 v3_complex `CURRENT_BASELINE` 仍是 mask-only grid decoder + forward consistency `lambda_forward=0.10` + validation-selected threshold `0.80`；但当前研究主线已经从 2D / quasi-2D profile-forward 小修切换到 true 3D / Piao-style geometry profile feasibility。

当前路线边界必须写清楚：

- 当前 COMSOL 支持真实 3D volume solve，并已验证 Bx / By / Bz 输出和 source `Je` 方向控制。
- 当前已有 rect / rotated / polygon / profile geometry 多数仍是 constant-depth prism / top-view extrusion。
- RBC / variable-depth true 3D profile generation 尚未验证，是下一阶段 smoke 的 blocker。
- 不要把 RBC 3D profile、depth-varying surface 或 true 3D train-ready pack 写成既成事实。
- dense mask baseline 只作为 comparator，不再作为当前 geometry-forward 主线。

每次任务开始前，优先读取：

1. `NEXT_STEP.md`
2. `PINN优化路线.md`
3. `EXPERIMENT_LOG.md`
4. `CURRENT_BASELINE.md`
5. `术语说明.md`
6. 与当前任务相关的 `results/summaries/`

如果文档与记忆或口头上下文不一致，以仓库内最新 Markdown、实验记录和实际结果文件为准。

## Markdown 文档管理规则

- 每次完成一个 prompt 前，必须检查本轮结果是否需要同步 Markdown：至少检查 `README.md`、`NEXT_STEP.md`、`EXPERIMENT_LOG.md`、`PINN优化路线.md`、`术语说明.md`、`CURRENT_BASELINE.md` 和任务相关 summary / baseline 文档。
- 如果本轮发生 baseline 更新，必须同步 `CURRENT_BASELINE.md`、`NEXT_STEP.md`、`EXPERIMENT_LOG.md`，并检查 README 是否需要改。
- 如果本轮发生路线级转向、阶段收口、研究主线切换或重要 blocker 结论，必须同步 `PINN优化路线.md`、`NEXT_STEP.md`、`EXPERIMENT_LOG.md`，并检查 `README.md` 和 `术语说明.md`。
- `README.md` 是项目入口页，只记录当前状态、baseline、主线阶段、文档导航和运行提醒；不要写成长流水账，且不能长期停留在旧阶段。
- `EXPERIMENT_LOG.md` 记录实验历史和阶段结论。
- `NEXT_STEP.md` 只记录最近下一步，不堆长期历史。
- `PINN优化路线.md` 记录路线级判断、停止方向和阶段切换原则。
- `术语说明.md` 记录新术语和关键定义；出现 true 3D / RBC / Piao-style / multi-axis / multi-direction 等新路线术语时要同步。
- 不要在每个小实验都大改 README；只有 baseline 更新、路线级转向、阶段级数据链路变化、关键 blocker 或下一阶段主线改变时才更新。

## Claude Code review 规则

- Codex 需要自行判断是否有必要调用 Claude Code review。
- 不要过于频繁调用 Claude Code；只有在变更涉及关键实验结论、baseline / benchmark、跨仓库数据生成、复杂脚本、可能影响项目路线，或 Codex 判断存在较高回归风险时才调用。
- 如果判断需要 review，Codex 应自行在终端中调用 Claude Code 完成 review，并根据 review 结果修复 must-fix 或确认无需修改。
- 最终输出中只需要简要说明是否调用 Claude Code review，以及结论或处理结果。
- 不要在输出结尾询问用户是否需要 review，也不要让用户另行手动发起 review。
- 重大 baseline 替换、benchmark 转换、可能影响项目方向的决策，仍需在执行前明确提醒用户并等待确认。

## Git 与数据安全

不要提交：

- `data/`
- `.npz`
- COMSOL `.mph`
- raw COMSOL CSV / generated raw data
- `checkpoints/`
- `*.pt` / `*.pth`
- `results/previews/`
- `*.png` / `*.jpg`
- `__pycache__/`
- `notes/`

可以提交：

- 明确允许的脚本；
- 明确要求记录的 `results/metrics/*.csv`；
- 明确要求记录的 `results/summaries/*.txt` / `*.md`；
- 当前任务允许提交的 Markdown。

提交前必须运行 `git status --short`。不要使用 `git add .`。如果 results 下记录文件被 `.gitignore` 忽略，只对明确允许的 txt/csv/md 使用 `git add -f`。

## 删除安全规则

禁止批量删除文件或目录。不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，一次只能删除一个明确路径的文件。如果需要批量删除文件，应停止操作，并请求用户手动删除或明确授权后再处理。

## 运行环境

默认使用以下 Python 解释器运行 PINN_project 脚本：

```powershell
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" <script.py>
```

不要默认用系统 `python` 或 `py`，除非任务明确要求或环境检查表明需要。

## 工作原则

1. test set 只用于阶段性最终评估，不用于调参。
2. checkpoint selection 和 threshold selection 必须使用 validation set。
3. 不要覆盖已有关键 checkpoint 或数据包。
4. 新实验使用独立脚本和独立输出路径。
5. 第 20.65 之后，除非用户明确要求，不再继续 2D profile-forward surrogate / refinement / observation 小修。
6. COMSOL_Multiphysics_MCP 是外部 forward generation 工程；PINN_project 只接收、检查、训练和记录其输出数据。

## 中文编写规则

除代码标识、命令、路径、字段名、日志原文和必要英文专有名词外，需求说明、计划、总结、验收标准、路线文档、review 记录和最终回复默认使用中文编写。
