# S112 COMSOL parametric inverse plan

## 目的

S112 将 COMSOL V2 的后续路线从 dense conditional mask loss 微调切换到 parametric inverse model。核心原因是 V2 dense mask runner 已多次在全背景、全前景和 localization 不足之间摆动，而 V2 数据中已有 `defect_params`，可以先学习低维结构化几何参数。

## 计划新增模块

- `comsol_parametric_targets.py`：从 V2 `defect_params.csv` 或 NPZ 中构造 component-level parametric targets。
- `comsol_parametric_inverse_models.py`：定义 `ParametricInverseNet`，从 flatten multi-height Bz signal 预测 component presence、type 和连续几何参数。
- `train_comsol_parametric_inverse.py`：运行首个小规模 supervised parametric inverse probe，并输出参数误差和可选 rasterized mask 指标。

## 边界

- 当前只做 skeleton、smoke test 和 small train probe。
- 不保存模型权重或 checkpoint。
- 不把 parametric route 写成主线替代结论。
- 第一版 rasterization 只用于评估，不反传。

## 下一步

- S113 构造 train / val / test parametric targets。
- S114 创建 parametric inverse model skeleton。
- S115 运行 V2 parametric inverse training probe。
- S116 汇总路线可行性和下一步。
