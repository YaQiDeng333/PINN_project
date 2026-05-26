# S307 Center-Anchored Y-Bin Repair Decision

S303-S307 完成了 y-bin failure diagnostics、default-off y-bin extra loss implementation 和 matched split quick gate。没有运行 multi-seed，没有生成新 COMSOL 数据，没有改模型结构，也没有替换 S185/S181 candidate。

## Decision

本阶段不通过 acceptance gate。`neighbor_soft_y` 说明 y-bin soft target 对机制有轻微正向作用：val/test y-bin acc、within-1 acc、zero-IoU 数量都有改善。但改善幅度不足，而且 test mean IoU 没有高于 same-run reference。`distance_soft_y` 更弱。

主结论：y localization only partially repaired, final mask still local-shape / conditioning limited。继续调 y loss 不符合 stop condition；下一阶段应转向 **local-shape conditioning / bounded local output**，因为当前 y-bin 变好后最终 mask IoU 没有同步稳定提升。

## Boundary

- 不进入 multi-seed。
- 不继续 y hard-weight sweep。
- 不扩大训练步数或模型容量。
- 不生成更多 COMSOL 数据。
- 不替换 S185/S181 center-bin candidate。
- 不写成 main baseline replacement。
