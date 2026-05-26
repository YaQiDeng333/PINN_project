# S48 conditional model skeleton summary

## 为什么进入 conditional model 阶段

S47 之后，per-sample semi-supervised optimization 阶段已经给出清晰边界：

- weak-form 双网络 runner 已经工程跑通；
- 半监督 `BCE mask prior` 在多个分辨率和样本规模下稳定优于 baseline；
- 200x100 下当前较稳配置仍是 `32x2 + area3_bce7`；
- `128x4` 有容量潜力，但样本级稳定性不足；
- 当前最强结果仍依赖 `mu_label` / `label_mask`，不能作为主线替代模型。

因此 S48 开始转向 `signal-conditioned dual network`：模型推理时输入 `Bz signal + coords`，输出 `mu_pred` / `phi_pred`，不依赖 `mu_label mask`。

## 新增文件

- `CONDITIONAL_DUAL_NETWORK_PLAN.md`：记录 conditional model 阶段动机、目标、结构和边界。
- `conditional_dual_models.py`：定义 `BzEncoder`、`ConditionalMLP`、`ConditionalMuNet`、`ConditionalPhiNet`、`ConditionalDualNet`。
- `smoke_test_conditional_dual_models.py`：验证模型 forward shape、`mu` 范围和错误 shape 的 `ValueError`。
- `experiments/dual_network/S48_conditional_model_skeleton/summary.md`：记录本阶段产物。

## smoke test 检查内容

`smoke_test_conditional_dual_models.py` 构造：

- `batch_size=3`
- `signal_len=200`
- `N=500` 随机 coords
- `signals` shape `[3,200]`
- `ConditionalDualNet(signal_len=200, latent_dim=32, hidden_dim=64, num_layers=3)`

检查：

- `coords` 为 `[N,2]` 时，`latent` shape 为 `[3,32]`，`mu` / `phi` shape 为 `[3,N,1]`；
- `coords` 为 `[B,N,2]` 时，输出 shape 相同；
- `mu` 范围保持在 `[1.0,1000.0]`；
- 错误的 `signals` / `coords` shape 会抛出清晰 `ValueError`。

## 当前结果边界

S48 没有训练，没有 `defect_iou`、`defect_area_pred`、`mu_mse` 或 `mu_mae` 指标。

本阶段只证明 conditional model 的最小 PyTorch 接口和 shape 约束成立。它还不能说明模型效果，也不能替代 per-sample runner 或主线 baseline。

## 下一步建议

S49 建议创建 conditional runner 的数据接口 / batch loader：

- 从 `.npz` 读取 `signals`、`coords`、`mu_label`；
- 组织 batch；
- 对接 `ConditionalDualNet`;
- 先跑极小 batch 的 supervised smoke train；
- 继续保持推理接口不使用 `mu_label` / `label_mask`。
