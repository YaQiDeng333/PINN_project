# S119 COMSOL refined parametric inverse training probe

## 数据

- V2 train / val / test NPZ 使用 S84 converted NPZ。
- targets 使用 S118 refined targets。

## 配置

### refined_mlp

- steps: `3000`
- lr: `1e-3`
- hidden_dim: `128`
- latent_dim: `64`
- max_components: `3`
- angle encoding: `sincos`
- continuous normalization: train stats from S118
- type class weighting: `inverse_freq`

### refined_mlp_longer

未运行。原因是 `refined_mlp` 的 val/test `param_mask_iou` 和 type accuracy 没有改善，继续加长预计信息增益低。

## 结果

| config | split | presence_acc | type_acc | continuous_mae | center_mae | axis_mae | rotation_mae | depth_mae | mask_iou | mask_dice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| refined_mlp | train | 1.000000e+00 | 1.000000e+00 | 8.696500e-04 | 2.997519e-04 | 1.208724e-05 | 2.569852e-01 | 8.376217e-06 | 6.689502e-01 | 7.994292e-01 |
| refined_mlp | val | 1.000000e+00 | 6.166667e-01 | 2.123421e-02 | 1.919508e-03 | 6.275431e-04 | 7.278932e+00 | 2.760201e-04 | 3.257646e-01 | 4.660925e-01 |
| refined_mlp | test | 1.000000e+00 | 6.833333e-01 | 2.402147e-02 | 1.523438e-03 | 5.717699e-04 | 7.859528e+00 | 3.198527e-04 | 3.885092e-01 | 5.348573e-01 |

## 与 S115 对比

- rotation MAE: val 从 `7.731843` 降到 `7.278932`，test 从 `7.740396` 升到 `7.859528`，整体没有稳定改善。
- type accuracy: val 从 `0.65` 降到 `0.6167`，test 从 `0.6667` 升到 `0.6833`，变化不稳定。
- mask IoU: val 从 `0.369908` 降到 `0.325765`，test 从 `0.424462` 降到 `0.388509`。

## 判断

S118 的 target refinement 改善了 target 语义表达，但本轮 MLP probe 没有改善 held-out mask IoU。当前不建议只靠更长 steps 继续扫；下一步应优先考虑 encoder / component head / loss 分解，而不是继续加长同一 MLP 配置。
