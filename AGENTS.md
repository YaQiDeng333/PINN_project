# AGENTS.md

## 项目执行规则

本项目是 MFL / 磁通泄漏缺陷边界形状反演项目。当前 v3_complex `CURRENT_BASELINE` 是 mask-only grid decoder + forward consistency `lambda_forward=0.10` + validation-selected threshold `0.80`。当前主线已经进入 COMSOL / forward data augmentation 阶段。

每次任务开始前，优先读取：

1. `CURRENT_BASELINE.md`
2. `NEXT_STEP.md`
3. `PINN优化路线.md`
4. `EXPERIMENT_LOG.md`
5. `术语说明.md`
6. 与当前任务相关的 `results/summaries/`

如果文档与记忆不一致，以仓库内文档和结果记录为准。

## Markdown 文档管理规则

- baseline 更新时，必须同步 `CURRENT_BASELINE.md`、`NEXT_STEP.md`、`EXPERIMENT_LOG.md`。
- 发生路线级转向时，必须同步 `PINN优化路线.md` 和 `README.md`。
- `README.md` 是项目入口页，只记录当前状态、baseline、主线阶段、文档导航和运行提醒；不要写成长流水账。
- `EXPERIMENT_LOG.md` 记录实验历史和阶段结论。
- `NEXT_STEP.md` 只记录最近下一步，不堆历史。
- `术语说明.md` 记录新术语和关键定义。
- 不要让 `README.md` 长期停留在旧阶段。
- 不要在每个小实验都大改 `README.md`；只有 baseline 更新、路线级转向或阶段级数据链路变化时才更新。

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
5. 第 20 阶段以后，除非用户明确要求，不再继续 decoder / loss / threshold / geometry / refinement 小变体。
6. COMSOL_Multiphysics_MCP 是外部 forward generation 工程；PINN_project 只接收、检查、训练和记录其输出数据。
