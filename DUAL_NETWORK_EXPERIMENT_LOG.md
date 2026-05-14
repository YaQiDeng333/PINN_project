# DUAL_NETWORK_EXPERIMENT_LOG

## 0. 支线定位

本日志记录 `feature/dual-network-variational` 支线的理论设计、代码骨架、最小闭环、数据接口和 weak-form 实验过程。

本支线目标：

- 探索基于 `phi-Net` / `mu-Net` 的双网络变分反演方法；
- 使用变分场重构和弱形式材料更新；
- 当前支线不替代 main，不声称优于主线；
- 当前阶段仍是方法验证和最小闭环实验。

## 1. 已完成 checkpoint 概览

1. `docs: add variational dual-network branch plan`
   - 建立支线理论方案；
   - 明确固定 `phi` 后不能直接用 data loss 更新 `mu`；
   - 改为 weak-form material update。

2. `docs: add dual-network implementation plan`
   - 明确新增文件结构；
   - 统一 `train_dual_variational.py`、`dual_network_models.py`、`dual_network_losses.py`、`evaluate_dual_variational.py`；
   - 明确第一版 `y_s = y_max = 10.0`。

3. `docs: audit data structure for dual-network branch`
   - 确认 `signals` 来自 `bz_signal[-1, :]`；
   - `x in [-15, 15]`，200 点；
   - `y in [0, 10]`，100 点；
   - `mu_maps` shape 为 `[N, 100, 200]`；
   - 数据可支持最小支线闭环，但物理正问题仍是解析近似。

4. `feat: add dual-network variational skeleton`
   - 新增 `PhiNet`、`MuNet`；
   - 新增 `energy_loss`、`data_loss`、`tv_loss`、`weak_form_loss` skeleton；
   - train / evaluate 文件仍为 skeleton。

5. `test: add dual-network smoke test`
   - 验证模型 import、前向传播、一阶梯度和基础 loss 可以运行。

6. `test: add minimal dual-network variational loop`
   - 使用 synthetic signal 跑通 `phi-step` / `mu-step` 交替优化；
   - 验证冻结 / 解冻和 autograd 路径。

7. `feat: add dual-network data utilities`
   - 支持 `.npz` 中 `coords` 或 `x/y` 两种坐标结构；
   - 确认 `x/y -> coords` 与 `mu_map` flatten 顺序一致。

8. `test: add minimal single-sample dual-network loop`
   - 将 `.npz` 数据接口接入单样本交替优化原型；
   - 无 `--npz-path` 时正常退出；
   - 有临时 `.npz` 时可跑通。

9. `test: add single-sample npz smoke test`
   - 固化临时 `.npz` 集成测试；
   - 验证单样本 `.npz` 闭环可重复运行。

10. `feat: add compact-support weak-form test gradients`
    - 新增 `generate_compact_support_test_grads`；
    - 使用局部 bump test function gradients；
    - smoke test 验证 `weak_form_loss` 可 backward 到 `MuNet`；
    - `PhiNet` 参数不接收 `weak_form_loss` 梯度。

## 2. 当前 weak-form 阶段记录

### Step S1: Compact-support weak-form test gradients

目的：

用第一版局部 bump 测试函数替代 dummy `test_grads`，为 `weak_form_loss` 提供真实的测试函数梯度输入。

方法：

- 使用 `v_q = (1 - r2)^2` 的 compact-support bump 函数；
- `r2 < 1` 内部有非零梯度；
- `r2 >= 1` 外部梯度为 0；
- 输出 `test_grads` shape 为 `[Q, N, 2]`。

验证：

- `smoke_test_weak_form_test_functions.py` 通过；
- `weak_form_loss` 返回 finite scalar；
- backward 后 `MuNet` 参数有梯度；
- `PhiNet` 参数没有梯度；
- `py_compile` 通过。

当前限制：

- `normalize=True` 会改变不同 support 尺寸测试函数的相对权重；
- 当前还没有真实 quadrature weights；
- 当前还没有 FEM-like basis functions；
- 当前还没有在真实单样本训练中替代 dummy `test_grads`；
- 当前不声称 weak-form 已经严格物理完备。

下一步：

- 将 `generate_compact_support_test_grads` 接入 `minimal_dual_single_sample_loop.py`；
- 用临时 `.npz` 跑单样本真实 weak-form 闭环；
- 记录 `loss_phi`、`loss_mu`、`mu_pred` 范围和是否稳定。

## 3. 当前不做的事情

- 不替换 main；
- 不改主线训练代码；
- 不进行大规模训练；
- 不声称当前方法优于主线；
- 不保存 checkpoint；
- 不输出论文结论；
- 不 push 到 GitHub。

## 4. 后续计划

1. 将真实 compact-support `test_grads` 接入 single-sample loop；
2. 添加对应 smoke test；
3. 在单样本上观察 `loss_phi` / `loss_mu` 是否稳定；
4. 再考虑小批量样本；
5. 最后再考虑正式训练脚本 `train_dual_variational.py`。
