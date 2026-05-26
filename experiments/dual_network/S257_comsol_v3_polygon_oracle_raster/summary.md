# S257 COMSOL V3 Polygon Oracle Raster Gate

The hard polygon rasterizer reconstructs the S254 masks from normalized polygon vertices using boolean OR union across present components.

## Oracle Metrics

| split | samples | mean IoU | min IoU | max IoU | mean Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | `30` | `1.000000` | `1.000000` | `1.000000` | `1.000000` |
| val | `10` | `1.000000` | `1.000000` | `1.000000` | `1.000000` |
| test | `10` | `1.000000` | `1.000000` | `1.000000` | `1.000000` |

Gate threshold was mean IoU `>=0.95` and per-sample min IoU `>=0.90`; all splits pass.

This confirms that the polygon target/rasterizer route removes the old `center + axis + rotation` oracle ceiling for this true-geometry V3 pack.
