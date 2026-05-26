# S109 COMSOL V2 localization-aware objective probe

## 执行状态

S109 已提前停止长实验搜索。本阶段只保留已经完成的三组结果：

- `bidir_margin_val_select`
- `bidir_margin_area_ratio`
- `bidir_margin_floor`

`direct_mask_area_ratio` 未运行。停止原因是前三组已经显示 margin + validation / area calibration 的信息增益有限，继续运行剩余长配置预计耗时高且收益低。

## 已完成结果

| config | split | defect_iou | defect_area_pred | mu_mse | mu_mae | best_step | best_eval_iou |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bidir_margin_val_select | train | 5.641042e-02 | 1.525398e+04 | 2.502306e+05 | 5.002301e+02 | 2000 | 5.671201e-02 |
| bidir_margin_val_select | val | 5.671201e-02 | 1.615885e+04 | 2.502399e+05 | 5.002395e+02 | 2000 | 5.671201e-02 |
| bidir_margin_val_select | test | 5.617056e-02 | 1.518510e+04 | 2.505939e+05 | 5.005896e+02 | 2000 | 5.671201e-02 |
| bidir_margin_area_ratio | train | 5.212874e-02 | 1.800313e+04 | 2.504842e+05 | 5.004815e+02 | 2000 | 5.467077e-02 |
| bidir_margin_area_ratio | val | 5.467077e-02 | 1.888105e+04 | 2.509796e+05 | 5.009744e+02 | 2000 | 5.467077e-02 |
| bidir_margin_area_ratio | test | 5.347047e-02 | 1.841155e+04 | 2.505764e+05 | 5.005737e+02 | 2000 | 5.467077e-02 |
| bidir_margin_floor | train | 1.508809e-01 | 7.240820e+03 | 2.163793e+05 | 4.613898e+02 | 3000 | 1.344480e-01 |
| bidir_margin_floor | val | 1.344480e-01 | 7.624900e+03 | 2.210390e+05 | 4.668041e+02 | 3000 | 1.344480e-01 |
| bidir_margin_floor | test | 1.440315e-01 | 7.415400e+03 | 2.181351e+05 | 4.633886e+02 | 3000 | 1.344480e-01 |

## 当前判断

- `bidir_margin_val_select` 的 validation-aware selection 选中 step 2000，但该 endpoint 更偏过大前景，IoU 低于 S105 `bidirectional_margin_lambda1`。
- `bidir_margin_area_ratio` 没有改善 localization，反而进一步接近全前景。
- `bidir_margin_floor` 是本阶段已完成配置中最好的一组，但仍明显低于 S85 `big_multichannel_v2`。
- 当前 V2 问题不是简单 margin + area + endpoint selection 能解决。

## 停止决定

不继续运行 `direct_mask_area_ratio`，后续所有新 V2 objective / output path 先进入 quick diagnostic gates，再决定是否运行 full V2 train / val / test。
