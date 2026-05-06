# AGENTS.md

## 项目规则

本项目是 PINN 缺陷边界形状反演 / 漏磁反演项目。

每次开始任务前，请先阅读：

1. README.md
2. PINN优化路线.md
3. NEXT_STEP.md
4. EXPERIMENT_LOG.md
5. CURRENT_BASELINE.md，如果存在

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
