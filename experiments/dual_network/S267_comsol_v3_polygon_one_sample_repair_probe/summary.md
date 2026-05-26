# S267 COMSOL V3 Polygon One-Sample Repair Probe

S267 only runs the one-sample polygon inverse repair matrix. The first candidate, `longer_overfit`, passes the gate, so the later grid-loss and area/edge variants are not run.

## Runs

| run | status | train IoU | vertex MAE | pred area | target area |
| --- | --- | ---: | ---: | ---: | ---: |
| `longer_overfit` | passed | `1.000000` | `7.786439e-07` | `189` | `189` |
| `grid_vertex_loss` | skipped | | | | |
| `grid_vertex_area_edge` | skipped | | | | |

## Diagnostic Comparison

| diagnostic | S262 failed run | S267 `longer_overfit` |
| --- | ---: | ---: |
| hard polygon IoU | `0.883178` | `1.000000` |
| area diff pixels | `25` | `0` |
| max abs x error cells | `0.333211` | `0.003405` |
| max abs y error cells | `0.494064` | `0.005855` |
| polygon area ratio in grid units | `1.043460` | `1.000522` |

## Decision

The one-sample failure was a precision/convergence issue exposed by hard raster sensitivity, not a polygon target or ordering bug. Because the old loss passes when given enough optimization steps, S267 does not require differentiable polygon raster loss before returning to the staged tiny-overfit sequence.
