# AGENTS.md

## 项目规则

本项目是 PINN 缺陷边界形状反演 / 漏磁反演项目。

每次开始任务前，请先阅读：

1. README.md
2. PINN优化路线.md
3. NEXT_STEP.md
4. EXPERIMENT_LOG.md
5. CURRENT_BASELINE.md，如果存在

## 每次任务开始前必须读取的项目文档

每次开始任何任务前，必须先阅读以下文件：

1. README.md
2. PINN优化路线.md
3. NEXT_STEP.md
4. EXPERIMENT_LOG.md
5. CURRENT_BASELINE.md，如果存在
6. 术语说明.md，如果存在
7. results/summaries/ 中与当前任务相关的 summary 文件，如果存在

阅读后再开始修改代码或执行命令。

如果任务涉及：

* 数据生成：还要阅读 data_generator_v2.py
* 训练：还要阅读 train_pinn.py
* 评估：还要阅读 evaluate_pinn.py
* 参数扫描：还要阅读 parameter_sweep.py
* Git 提交：还要先检查 .gitignore 和 git status

每次完成阶段性任务后，必须检查是否需要更新：

1. README.md
2. PINN优化路线.md
3. NEXT_STEP.md
4. EXPERIMENT_LOG.md
5. CURRENT_BASELINE.md

不要每次要求用户重复提醒。

## 文档同步规则

每次完成以下任意情况后，必须检查并更新 README.md：

1. 新增或删除主要脚本；
2. 新增模型训练流程；
3. 新增评价指标；
4. 新增重要结果文件；
5. 改变当前推荐模型；
6. 改变当前 baseline；
7. 完成一个阶段步骤；
8. 修改 results/ 或 checkpoints/ 的推荐使用方式。

每次实验完成后，必须追加更新 EXPERIMENT_LOG.md，不要覆盖旧记录。

每次阶段完成后，必须更新：

1. PINN优化路线.md
2. NEXT_STEP.md
3. README.md
4. CURRENT_BASELINE.md，如果当前最佳模型发生变化

## 运行环境

默认使用以下 Python 解释器运行脚本：

& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" 脚本名.py

不要直接使用 python 或 py。

## 工作原则

1. 不要覆盖已有关键模型文件；
2. 新实验模型使用新的文件名；
3. test 集只用于阶段性最终评估，不要频繁用 test 集调参；
4. 主要基于 val 集选择参数；
5. 不要删除 results/ 或 checkpoints/ 中的文件，除非用户明确要求；
6. 修改代码后说明修改了哪些文件、生成了哪些结果、当前推荐模型是否变化。
