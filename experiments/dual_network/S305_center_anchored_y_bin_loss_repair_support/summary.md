# S305 Center-Anchored Y-Bin Loss Repair Support

S305 只修改 center-anchored polygon runner 的 loss 支持，不改模型结构、不改 target schema、不改变默认行为。

新增 CLI：

- `--center-y-bin-extra-loss-mode none|neighbor_soft_ce|distance_soft_ce`，默认 `none`。
- `--lambda-center-y-bin-extra`，默认 `0.0`。
- `--center-y-bin-neighbor-smoothing`，默认 `0.0`。
- `--center-y-bin-distance-sigma`，默认 `0.75`。

实现语义：

- 原 `center_bin_loss = 0.5 * (xCE + yCE)` 保持不变。
- 新 y-bin loss 是额外项，只在显式设置 non-none mode 且 `lambda_center_y_bin_extra > 0` 时启用。
- `neighbor_soft_ce` 给 true y-bin 保留主质量，将 smoothing 质量分配给相邻 bin，边界处重新归一。
- `distance_soft_ce` 用 y-bin 距离的 Gaussian soft target，给分类头注入空间邻近偏置。
- history、metrics 和 run summary 记录 extra loss、weighted extra loss、y-bin abs error 和 within-1 accuracy。

验证：

- runner smoke 覆盖默认路径和 `neighbor_soft_ce` 路径。
- y-bin diagnostics smoke 覆盖 CSV join、histogram 和 summary 输出。
- `py_compile` 覆盖 touched scripts。
