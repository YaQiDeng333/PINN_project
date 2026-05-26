# S156 COMSOL type / rotation loss stage summary

## 背景

S151-S155 已经把 learned forward consistency 和直接 type / rotation extra loss 都做了 quick diagnostic。当前稳定 best 仍是 S115 / S143 / S149 / S154 `param_only_reference` raw MLP / shared head / fixed-order parameter-only baseline。

## S152 结论

- forward residual 对 `rotation` perturbation 非常敏感。
- forward residual 对 `type` swap 中等敏感。
- forward residual 对 `axis` scaling 基本不敏感。
- predicted residual 与最终 mask IoU 的相关性有限，因此更适合作为 diagnostic / ranking，而不是当前默认训练 loss。

## S154 结论

- `type_extra` 没有改善 type accuracy，val mask IoU 明显下降，test 的微小提升不稳定。
- `rotation_extra` 有局部信号，改善 train rotation fit 和 val mask IoU，但 test 低于 S115 baseline。
- `type_rotation_extra` 没有形成新默认，test mask IoU 明显下降。
- 当前最佳仍是 `param_only_reference`。

## 当前判断

不再继续盲扫 type / rotation loss 权重，也不回到 dense conditional mask runner。下一步需要用 parameter-level oracle ablation 从已有 predictions 出发，逐项替换 GT type、rotation、center、axis、depth 和 continuous，直接判断哪些参数误差真正限制 final rasterized mask IoU。

## 自评

- S156 只总结既有阶段，不运行训练。
- 结论没有声称 parametric route 已成功，只说明继续 loss 权重微调的信息密度不足。
- 下一步转向 oracle ablation diagnostic，符合当前瓶颈定位目标。
