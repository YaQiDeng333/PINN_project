# S114 COMSOL parametric inverse model skeleton

## 模型结构

S114 新增 `comsol_parametric_inverse_models.py`，包含：

- `ParametricBzEncoder`：MLP encoder，输入 flattened multi-height Bz signal `[B,600]`，输出 latent `[B,latent_dim]`。
- `ComponentParamHead`：从 latent 预测 component-level outputs。
- `ParametricInverseNet`：encoder + head 的包装模型。

## 输入输出 shape

默认配置：

- input signals: `[B,600]`
- `presence_logits`: `[B,max_components]`
- `presence_prob`: `[B,max_components]`
- `type_logits`: `[B,max_components,num_types]`
- `continuous`: `[B,max_components,num_continuous]`

当前 S113 target schema 使用 `max_components=3`、`num_continuous=6`。

## 当前边界

- 这只是 skeleton，不保存权重。
- runner 负责将 `[B,3,200]` signals flatten 为 `[B,600]`。
- 下一步 S115 使用该模型跑首个 parametric inverse training probe。
