# S235 COMSOL V3 repaired oracle rasterization gate

S235 reruns the hard parametric rasterizer on S233 converted repaired V3 data and S234 parametric targets.

| split | samples | avg oracle IoU | min oracle IoU | max oracle IoU | avg oracle Dice | avg abs area diff |
|---|---:|---:|---:|---:|---:|---:|
| train | 30 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| val | 10 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| test | 10 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

Gate result: pass. Oracle IoU is not below the `0.95` stop threshold on any split.
