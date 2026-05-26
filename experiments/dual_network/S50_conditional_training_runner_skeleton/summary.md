# S50 conditional training runner skeleton summary

## 目的

S50 创建 signal-conditioned dual-network 的最小 supervised training runner。目标是验证一个统一模型可以接收 `signals + coords`，并通过监督 mask loss 完成最小训练闭环。

本阶段仍是 skeleton / smoke test，不是正式训练实验。

## 新增 runner

`train_conditional_dual.py` 支持：

- `--npz-path`
- `--output-dir`
- `--sample-indices`
- `--device`
- `--steps`
- `--lr`
- `--hidden-dim`
- `--num-layers`
- `--latent-dim`
- `--lambda-mask-bce`
- `--lambda-mask-dice`
- `--lambda-mu-mse`
- `--mask-temperature`

如果未提供 `--npz-path` 或 `--output-dir`，runner 只打印说明并正常退出。

## 训练输入输出

输入来自 `conditional_dual_data_utils.py`：

- `signals [B, signal_len]`
- `coords [N,2]`
- `mu_label [B,N,1]`
- `mask_label [B,N,1]`

模型输出来自 `ConditionalDualNet`：

- `mu [B,N,1]`
- `phi [B,N,1]`

S50 训练结束后只保存：

- `metrics.csv`
- `run_summary.md`

不保存模型权重、checkpoint、`.npy` 或图片。

## 当前 loss 组成

S50 使用 supervised mask losses：

- `mask BCE loss`
- `mask Dice loss`
- optional `mu_mse loss`

总损失：

`loss = lambda_mask_bce * bce_loss + lambda_mask_dice * dice_loss + lambda_mu_mse * mu_mse_loss`

当前没有接入 weak-form / physics loss。S50 的目标只是验证 signal-conditioned supervised training closure。

## smoke test

`smoke_test_train_conditional_dual.py` 使用 `tempfile.TemporaryDirectory()` 创建临时 `.npz` 和临时输出目录，不在项目目录留下数据。

测试内容：

- 调用 `train_conditional_dual.py --steps 5`；
- 检查 return code 为 0；
- 检查 `metrics.csv` 和 `run_summary.md` 存在；
- 检查 `metrics.csv` 有 3 行；
- 检查包含 `defect_iou`、`mu_mse`、`mu_mae`；
- 检查没有生成 `.pt`、`.pth`、`.ckpt` 或 `.npy`。

## 当前边界

S50 没有正式实验指标，不能与主线 baseline 比较。

当前 runner 是 conditional supervised skeleton，仍使用 `mu_label` / `mask_label` 作为训练监督，但推理 forward 接口只需要 `signals + coords`。

## 下一步建议

S51 建议使用真实小型 `.npz` 运行 conditional supervised runner，验证它能否在 train samples 上学习 mask，并记录最小 train-set 指标。之后再考虑 validation split、weak-form loss 和主线 baseline 对比。
