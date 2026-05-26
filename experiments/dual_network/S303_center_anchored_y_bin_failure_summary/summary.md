# S303 Center-Anchored Polygon Y-Bin Failure Summary

本阶段进入 center-anchored polygon y-bin localization repair。S298-S302 的 matched-coverage split 已排除“原 split 完全无覆盖”作为唯一解释：val/test 20/20 个样本都满足 component bin 到 train 的距离 <= 1，但 held-out mask IoU 仍很低。

关键现象：

- S300 train mean/min IoU = `0.995598 / 0.969697`，train x/y bin acc = `1.000000 / 1.000000`。
- S300 val/test mean/min IoU = `0.037245 / 0.000000` 和 `0.072368 / 0.000000`。
- S300 val/test zero-IoU = `8/10` 和 `9/10`。
- S301 诊断显示 `17/17` 个 zero-IoU held-out 样本仍有 y-bin 错误，`9/17` 同时有 x-bin 错误。

当前判断：根因更接近 y-bin localization 作为有序空间量的泛化失败，而不是 polygon oracle、signal、target decode 或训练集拟合失败。原 hard CE 把相邻 y-bin 错和远距离 y-bin 错都当作普通类别错误，不能表达 y 方向空间邻近关系；offset 又绑定 true bin 学习，wrong-bin decode 后无法稳定补偿。

边界：

- 不运行 multi-seed。
- 不加训练步数，不扩大模型。
- 不生成新 COMSOL 数据。
- 不替换 S185/S181 center-bin candidate。
- 不写成 main baseline replacement。
