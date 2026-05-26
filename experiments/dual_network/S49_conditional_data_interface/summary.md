# S49 conditional data interface summary

## 目的

S49 从 S48 的 `signal-conditioned dual-network` model skeleton 推进到最小 batch data interface。目标是让后续 conditional runner 可以从 `.npz` 中读取 `signals`、`coords`、`mu_label` 和 `mask_label`，并将它们以 batch 形式送入 `ConditionalDualNet`。

本阶段仍不做正式训练，不保存 checkpoint，也不声称模型效果。

## S48 review 小修

- `_build_tanh_mlp` 中的所有 `nn.Linear` 已使用 Xavier uniform 初始化。
- Linear bias 如果存在，则初始化为 0。
- `smoke_test_conditional_dual_models.py` 增加 backward smoke test，确认 `out_2d["mu"].sum() + out_2d["phi"].sum()` 可以反向传播，并且至少一个模型参数产生非 `None` gradient。

## 新增 data utilities

`conditional_dual_data_utils.py` 提供：

- `load_conditional_npz(npz_path)`：读取 `.npz`，要求包含 `signals`、`mu_maps`，并支持 `coords` 或 `x + y` 坐标字段。
- `build_coords_from_xy(x, y)`：使用 `np.meshgrid(indexing="xy")` 生成 `[ny*nx,2]` 坐标，flatten 顺序与 `mu_map.reshape(-1,1)` 对齐。
- `get_conditional_batch(dataset, sample_indices, device="cpu")`：返回 batch dict。
- `infer_signal_len(dataset)`：返回 `signals.shape[-1]`。

`get_conditional_batch` 输出：

- `signals`: `torch.float32`, shape `[B, signal_len]`
- `coords`: `torch.float32`, shape `[N,2]`
- `mu_label`: `torch.float32`, shape `[B,N,1]`
- `mask_label`: `torch.float32`, shape `[B,N,1]`
- `x_unique`: `torch.float32`
- `y_unique`: `torch.float32`

`coords` 默认不设置 `requires_grad_(True)`；后续如果 weak-form derivative 需要坐标导数，应由训练脚本显式设置。

## smoke test 覆盖

`smoke_test_conditional_dual_data_utils.py` 使用 `tempfile.TemporaryDirectory()` 创建临时 `.npz`，不在项目目录留下数据。

覆盖内容：

- `load_conditional_npz`
- `get_conditional_batch(dataset, [0,1,2])`
- `signals` shape `[3,20]`
- `coords` shape `[200,2]`
- `mu_label` / `mask_label` shape `[3,200,1]`
- `mask_label` 包含非零缺陷点
- `infer_signal_len(dataset) == 20`
- `ConditionalDualNet(signal_len=20, latent_dim=16, hidden_dim=32, num_layers=2)` forward
- 缺 `signals`、缺 `mu_maps`、缺 `coords` 且缺 `x/y` 时抛出 `ValueError`

## 当前边界

S49 没有正式训练，没有 `defect_iou`、`defect_area_pred`、`mu_mse` 或 `mu_mae` 指标。

当前只说明 conditional model 已具备最小 batch data interface，尚不能与主线 baseline 比较。

## 下一步建议

S50 建议创建 conditional supervised training runner skeleton：

- 使用 `conditional_dual_data_utils.py` 读取 batch；
- 使用 `ConditionalDualNet` 前向；
- 先实现极小 batch 的 supervised `mu_label` / `mask_label` loss smoke train；
- 不保存 checkpoint；
- 保持推理接口只依赖 `signals + coords`。
