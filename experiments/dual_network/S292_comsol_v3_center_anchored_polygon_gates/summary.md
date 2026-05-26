# S292 COMSOL V3 Center-Anchored Polygon Gates

S292 runs strict stop-on-fail gates: one-sample, five-sample, then train30. The same center-anchored configuration is used throughout: `steps=10000` for one/five sample and `steps=20000` for train30, `lr=1e-3`, `hidden_dim=128`, `latent_dim=64`, `lambda_center_offset=10.0`, `lambda_local_vertex=1.0`.

## Gate Matrix

| gate | status | mean IoU | min IoU | decoded vertex MAE | presence/type | x/y bin acc | out-of-grid | signed flips |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_sample | passed | `1.000000` | `1.000000` | `4.029769e-06` | `1.000000` / `1.000000` | `1.000000` / `1.000000` | `0` | `0` |
| five_sample | passed | `0.991549` | `0.957746` | `1.353233e-05` | `1.000000` / `1.000000` | `1.000000` / `1.000000` | `0` | `0` |
| train30 | passed train gate | `0.989276` | `0.857143` | `4.337009e-06` | `1.000000` / `1.000000` | `1.000000` / `1.000000` | `0` | `0` |

## Held-Out Observation

| split | mean IoU | min IoU | decoded vertex MAE | center x/y bin acc | zero IoU | out-of-grid | signed flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | `0.072402` | `0.000000` | `4.916527e-03` | `0.461538` / `0.153846` | `8` | `0` | `0` |
| test | `0.084416` | `0.000000` | `1.655222e-03` | `0.769231` / `0.076923` | `8` | `0` | `0` |

The center-anchored train gate passes strongly and removes held-out signed-area flips / out-of-grid vertices in this run. Held-out IoU remains weak, mainly because center-bin accuracy is poor on val/test, especially y-bin accuracy.
