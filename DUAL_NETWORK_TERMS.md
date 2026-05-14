# DUAL_NETWORK_TERMS

## 双网络变分支线术语说明

本文档定义 `feature/dual-network-variational` 支线中反复出现的术语。该支线不替代 `main`，当前最稳定结果来自半监督 BCE mask prior；这些结果是 diagnostic upper bound，不是纯 weak-form 无监督反演成功。

## 1. phi-Net

`phi-Net` 是输入坐标 `(x, y)`、输出磁标量势 `phi(x, y)` 的神经网络。它负责场重构。当前支线用能量项、边界/数据项等训练它。

## 2. mu-Net

`mu-Net` 是输入坐标 `(x, y)`、输出磁导率 `mu(x, y)` 的神经网络。它负责材料分布更新。当前实现将 raw 输出映射到 `[mu_min, mu_max]`。

## 3. weak-form loss

`weak-form loss` 来自静磁方程 `div(mu grad phi)=0` 的弱形式。固定 `phi` 后，用测试函数梯度 `grad(v_q)` 计算 `integral mu grad(phi) dot grad(v_q) dOmega` 的残差。它是支线核心物理项，但当前版本不能单独完成稳定缺陷定位。

## 4. compact-support test functions

局部紧支撑测试函数。当前第一版使用 bump 函数，只在给定 center 和 radius 附近非零，用来构造 weak-form residual。后续可替换为 FEM-like basis 或更严格的积分权重。

## 5. test_grads

`test_grads` 是测试函数梯度 `grad(v_q)`，形状通常为 `[Q, N, 2]`，其中 `Q` 是测试函数数量，`N` 是积分点数量。它不是可训练参数。

## 6. center_mode

`center_mode` 定义 compact-support 测试函数中心的布点策略。当前包括固定中心 `three/five/nine`，以及诊断用的 `signal_*`、`label_*` 模式。

## 7. test_radius

`test_radius` 是 compact-support 测试函数的半径。它影响 weak-form residual 的覆盖区域和数值尺度。S4 中 `5.0` 是较平衡的默认值，但不是理论最优值。

## 8. area prior

`area prior` 是用 soft defect fraction 约束预测缺陷面积的轻量项。它可以抑制全域低 `mu` 塌陷，但不能单独解决定位问题。它使用 `mu_label` 的面积信息，因此是诊断/半监督项。

## 9. mask prior / soft Dice prior

`mask prior` 或 soft Dice prior 使用 `mu_label < 500` 得到 label mask，并对 soft predicted defect mask 计算 Dice loss。它比 area prior 更局部，但仍不能完全抑制 false positives。

## 10. BCE mask prior

`BCE mask prior` 对 soft predicted defect mask 和 label mask 计算 binary cross entropy。S14-S19 中它是最有效的 false-positive 抑制信号。因为它使用 `mu_label` mask，所以是半监督/诊断上界，不是无监督反演方法。

## 11. baseline

在 S15 之后，`baseline` 通常指 runner 中的 `weak-form + area prior + soft Dice mask prior`，但不启用 BCE mask prior，即 `lambda_mask_bce_prior=0.0`。

## 12. bce

`bce` 通常指在 baseline 基础上启用 `lambda_mask_bce_prior=1.0` 的半监督/诊断运行。

## 13. defect_area_pred

`defect_area_pred` 是阈值 `mu_pred < 500` 的预测缺陷点数。它衡量预测低磁导率区域面积。

## 14. defect_area_label

`defect_area_label` 是阈值 `mu_label < 500` 的真实缺陷点数。它来自监督标签，只用于诊断或半监督 prior。

## 15. defect_iou

`defect_iou` 是预测缺陷 mask 与 label mask 的 intersection-over-union。阈值当前为 `mu < 500`。它是当前最直观的定位诊断指标。

## 16. mu_mse / mu_mae

`mu_mse` 和 `mu_mae` 分别是 `mu_pred` 与 `mu_label` 的均方误差和平均绝对误差。当前主要作为诊断指标，不代表核心无监督训练目标。

## 17. coords

`coords` 是定义域内部点坐标，形状通常为 `[N, 2]`。支线中计算 `phi`、`mu`、梯度和 weak-form residual 都依赖它。

## 18. probe_coords

`probe_coords` 是探头线坐标。第一版使用 `y_s=10.0`，与数据生成中 `bz_signal[-1, :]` 对齐。

## 19. bz_meas

`bz_meas` 是探头线上的测量/生成信号，在当前 `.npz` 中来自 `signals`。第一版把它用于 `data_loss`，不自动外推到 `lift_off` 域外位置。

## 20. mu_label

`mu_label` 是数据集中 `mu_maps` 展平后的真实磁导率标签。它用于诊断、可视化、半监督 prior 和上界实验，不是纯 weak-form 无监督训练的可用信息。

## 21. semi-supervised / diagnostic upper bound

半监督/诊断上界指使用了 `mu_label` 或 label mask 信息的实验。它能判断当前结构在有局部监督信号时的潜力，但不能证明无监督 weak-form 方法本身成功。

## 22. S18 / S19 / S20

- S18：20 个 `20x10` 样本的 runner probe。BCE 明显优于 baseline，验证了半监督上界的稳定性。
- S19：50 个 `20x10` 样本的 runner validation。BCE 继续稳定优于 baseline，是当前低分辨率下最强证据。
- S20：20 个 `40x20` 样本的 resolution probe。BCE 平均优于 baseline，但改善弱于低分辨率，不能说明高分辨率已经稳定有效。

## Boundary Notes

- 当前支线不替代 `main`。
- 当前不能声称纯 weak-form 无监督反演成功。
- `label-informed centers` 是 oracle diagnostic，只用于判断 center localization 是否是瓶颈。
- `BCE mask prior` 使用 `mu_label < 500`，因此属于半监督/诊断上界。
- 若继续推进，推荐方向是半监督双网络路线或设计 label-free 的局部化/false-positive 抑制机制。
