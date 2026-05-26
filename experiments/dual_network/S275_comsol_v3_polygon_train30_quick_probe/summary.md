# S275 COMSOL V3 Polygon Train30 Quick Probe

S275 runs the polygon inverse model on the full S254-S258 polygon V3 pack: train `[30,3,200]`, val `[10,3,200]`, and test `[10,3,200]`.

## Aggregate Metrics

| split | mean IoU | min IoU | presence acc | type acc | vertex MAE | pred area mean | target area mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.731445` | `0.518519` | `1.000000` | `1.000000` | `1.793932e-04` | `164.466667` | `143.666667` |
| val | `0.033122` | `0.000000` | `0.933333` | `0.692308` | `9.746788e-03` | `1688.000000` | `140.200000` |
| test | `0.089484` | `0.000000` | `0.966667` | `0.923077` | `5.539294e-03` | `666.700000` | `139.600000` |

## Gate

The train30 gate fails because train mean/min IoU are below the required `0.90` / `0.80`. Presence/type are solved on train, so the blocker is multi-sample vertex precision and raster-area alignment, not component existence or type classification.

No multi-seed, train30 extension, loss sweep, or architecture change is run in this stage.
