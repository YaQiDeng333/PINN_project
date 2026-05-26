# S110 COMSOL V2 localization objective early stop and diagnostic gate switch

## S108 代码状态

S108 validation-aware endpoint selection 已接入 `train_conditional_dual.py`：

- 默认 `val_selection_metric=none` 时保持旧行为。
- `eval_iou` / `eval_loss` 只在内存中保留 best state，不保存权重或 checkpoint。
- Claude Code review 已调用；must-fix 的 `val_selection_interval <= 0` guard 已修复。
- shared coords `[N,2]` 和 labels `[B,N,1]` 的 batch shape 已对照 `conditional_dual_data_utils.py` 确认。

## 为什么中止长实验

S109 原计划比较 validation-aware selection、margin + area calibration、direct mask + area loss 四组长实验。当前已完成的三组显示：

- `bidir_margin_val_select` 的 best endpoint 选在 step 2000，但结果更偏过大前景，val/test IoU 没有改善。
- `bidir_margin_area_ratio` 没有提升 localization，预测面积更接近全图。
- `bidir_margin_floor` 是已完成配置中最好的一组，但仍低于历史 S85 `big_multichannel_v2`。

因此继续运行 `direct_mask_area_ratio` 的信息增益预计有限，且长实验耗时高。本阶段切换到 quick diagnostic gate 模式。

## 已完成配置结果

| config | train IoU | val IoU | test IoU | train area | val area | test area | best_step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bidir_margin_val_select | 5.641042e-02 | 5.671201e-02 | 5.617056e-02 | 1.525398e+04 | 1.615885e+04 | 1.518510e+04 | 2000 |
| bidir_margin_area_ratio | 5.212874e-02 | 5.467077e-02 | 5.347047e-02 | 1.800313e+04 | 1.888105e+04 | 1.841155e+04 | 2000 |
| bidir_margin_floor | 1.508809e-01 | 1.344480e-01 | 1.440315e-01 | 7.240820e+03 | 7.624900e+03 | 7.415400e+03 | 3000 |

`direct_mask_area_ratio` 未运行，因阶段策略调整而停止。

## 当前结论

- V2 当前不是简单 area / margin / endpoint selection 能解决。
- 关键问题是模型没有稳定学到 localization / shape。
- 后续不再直接运行 full V2 train=100 / val=20 / test=20 长实验，除非先通过 quick gates。

## 后续 gate 规则

1. Gate 1: 5-sample train-overfit gate。
2. Gate 2: 20-train / 5-val mini generalization gate。
3. Gate 3: full V2 train / val / test。

只有前一 gate 通过，才允许进入下一 gate。
