# S304 Center-Anchored Polygon Y-Bin Diagnostics

本诊断只读取 S300 matched-coverage prediction export，没有运行训练。

## Split Summary

| split | samples | zero_iou | x_bin_acc | y_bin_acc | y_within1 | mean_abs_y_error | mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `30` | `0` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.995598` |
| val | `10` | `8` | `0.230769` | `0.230769` | `0.461538` | `1.769231` | `0.037245` |
| test | `10` | `9` | `0.583333` | `0.083333` | `0.333333` | `1.916667` | `0.072368` |

## Findings

- heldout zero-IoU present components: `22`
- zero-IoU components with y-bin error: `21`
- zero-IoU components with x-bin error: `13`
- heldout y-bin adjacent errors: `6`
- heldout y-bin distance >=2 errors: `15`

结论：held-out y-bin 错不是单纯相邻边界噪声，distance >= 2 的错误占多数；这支持 S305 使用 distance-aware / neighbor-aware soft target，而不是只加 y hard CE 权重。
