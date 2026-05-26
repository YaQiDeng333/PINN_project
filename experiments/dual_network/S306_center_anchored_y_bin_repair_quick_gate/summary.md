# S306 Center-Anchored Y-Bin Repair Quick Gate

本阶段只在 S299 matched-coverage split 上运行 quick gate，没有生成新 COMSOL 数据，没有改模型结构。所有命令使用 `steps=20000`、`seed=1`、`--export-predictions`，且不保存权重。

## Runs

| run | split | mean IoU | min IoU | x-bin acc | y-bin acc | y-bin abs err | y-bin within1 | zero-IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_reference | train | `0.995598` | `0.969697` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `0` |
| current_reference | val | `0.037245` | `0.000000` | `0.230769` | `0.230769` | `1.769231` | `0.461538` | `8` |
| current_reference | test | `0.072368` | `0.000000` | `0.583333` | `0.083333` | `1.916667` | `0.333333` | `9` |
| neighbor_soft_y | train | `0.988315` | `0.907895` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `0` |
| neighbor_soft_y | val | `0.056407` | `0.000000` | `0.307692` | `0.307692` | `1.384615` | `0.692308` | `7` |
| neighbor_soft_y | test | `0.069911` | `0.000000` | `0.833333` | `0.166667` | `1.666667` | `0.500000` | `8` |
| distance_soft_y | train | `0.991753` | `0.925000` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `0` |
| distance_soft_y | val | `0.017919` | `0.000000` | `0.230769` | `0.153846` | `1.692308` | `0.538462` | `9` |
| distance_soft_y | test | `0.068354` | `0.000000` | `0.666667` | `0.083333` | `2.000000` | `0.333333` | `9` |

## Gate Result

`current_reference` exactly reproduces S300 scale, so there is no reproduce risk. `neighbor_soft_y` improves val/test y-bin acc from `0.230769 / 0.083333` to `0.307692 / 0.166667`, reduces zero-IoU from `8/10 / 9/10` to `7/10 / 8/10`, and lowers y-bin abs error. However, the y-bin acc gains do not meet the planned thresholds, and test mean IoU drops slightly from `0.072368` to `0.069911`.

`distance_soft_y` does not improve the mechanism and worsens val IoU. Therefore S306 does not pass the y-bin repair acceptance gate.
