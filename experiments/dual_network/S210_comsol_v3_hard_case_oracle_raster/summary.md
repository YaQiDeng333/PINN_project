# S210 COMSOL V3 hard-case oracle raster summary

S210 rasterized the S209 ground-truth parametric targets back to masks and compared them with the S208 target masks. This is an ingest gate only; no training was run.

## Oracle Metrics

| split | samples | avg oracle IoU | min oracle IoU | avg Dice | avg target area | avg raster area | avg abs area diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 30 | 1.000000 | 1.000000 | 1.000000 | 204.3 | 204.3 | 0.0 |
| val | 10 | 1.000000 | 1.000000 | 1.000000 | 203.4 | 203.4 | 0.0 |
| test | 10 | 1.000000 | 1.000000 | 1.000000 | 204.0 | 204.0 | 0.0 |

## Gate

The oracle rasterization gate passes. All splits are far above the `0.70` threshold, and `gt` parametric masks align exactly with the provided masks for this fallback pilot.

## Boundary

The perfect oracle result confirms schema and rasterizer alignment for the single rectangular Block fallback pack. It does not validate true rotated or multi-component COMSOL geometry, because those solved cases are not yet present in this pack.
