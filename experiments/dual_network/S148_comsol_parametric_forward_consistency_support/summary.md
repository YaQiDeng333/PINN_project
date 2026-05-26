# S148 COMSOL parametric inverse forward-consistency support

## 目的

S148 在 S147 forward surrogate gate 通过后，新增 in-memory learned forward consistency inverse runner。

## 实现

- `train_comsol_parametric_inverse_forward_consistency.py`
  1. 读取 V2 signals 和 S113 parametric targets；
  2. 在内存中训练 geometry -> normalized Bz forward surrogate；
  3. 冻结 forward surrogate；
  4. 训练 inverse model；
  5. inverse loss = 原 parametric loss + `lambda_forward_consistency * forward_signal_mse`。
- predicted geometry vector 使用：
  - straight-through hard `presence`
  - straight-through hard one-hot `type`
  - 反归一化后的 predicted continuous geometry
- forward surrogate 和 consistency target 使用同一 train-only signal z-score stats。

## 边界

- 不保存 forward surrogate 权重。
- 不保存 inverse 权重。
- 不保存 checkpoint。
- 当前只用于 quick diagnostic probe，不替代 COMSOL solver。

## 自评

- Forward consistency loss 可通过 frozen surrogate 反传到 inverse geometry outputs。
- Geometry vector schema 与 forward pretrain 阶段一致。
- 后续 S149 需要比较 param-only baseline 与不同 forward consistency 权重。
