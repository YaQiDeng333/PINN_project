# S218 COMSOL V3 geometry normalized copy

S218 creates a normalized copy of the S208 V3 hard-case pack. It does not modify S208 raw or converted files.

## Transform

- `x_norm = (x_raw - 2250.0) * (0.08 / 4500.0)`
- `y_norm = (y_raw - 1500.0) * (0.02 / 3000.0)`
- `defect_center_x/y` use the same affine transform.
- `defect_axis_x *= 0.08 / 4500.0`
- `defect_axis_y *= 0.02 / 3000.0`
- `signals`, `mu_maps`, and `masks` are unchanged.
- depth / z fields are not scaled; the raw values are retained because the current hard rasterizer does not use depth for mask generation and V3 z convention has not been separately audited.

## Raw And Normalized Ranges

| split | x raw | x normalized | y raw | y normalized |
|---|---:|---:|---:|---:|
| train | `[0.000000, 4500.000000]` | `[-0.040000, 0.040000]` | `[0.000000, 3000.000000]` | `[-0.010000, 0.010000]` |
| val | `[0.000000, 4500.000000]` | `[-0.040000, 0.040000]` | `[0.000000, 3000.000000]` | `[-0.010000, 0.010000]` |
| test | `[0.000000, 4500.000000]` | `[-0.040000, 0.040000]` | `[0.000000, 3000.000000]` | `[-0.010000, 0.010000]` |

| split | center_x raw | center_x normalized | center_y raw | center_y normalized |
|---|---:|---:|---:|---:|
| train | `[704.070352, 3770.402010]` | `[-0.027483, 0.027029]` | `[810.000000, 2342.424242]` | `[-0.004600, 0.005616]` |
| val | `[781.407035, 3418.090452]` | `[-0.026108, 0.020766]` | `[810.000000, 1856.060606]` | `[-0.004600, 0.002374]` |
| test | `[1066.331658, 3770.402010]` | `[-0.021043, 0.027029]` | `[1200.000000, 2340.909091]` | `[-0.002000, 0.005606]` |

| split | axis_x raw | axis_x normalized | axis_y raw | axis_y normalized |
|---|---:|---:|---:|---:|
| train | `[420.000000, 880.000000]` | `[0.007467, 0.015644]` | `[110.000000, 264.000000]` | `[0.000733, 0.001760]` |
| val | `[450.000000, 880.000000]` | `[0.008000, 0.015644]` | `[120.000000, 260.000000]` | `[0.000800, 0.001733]` |
| test | `[420.000000, 850.000000]` | `[0.007467, 0.015111]` | `[110.000000, 264.000000]` | `[0.000733, 0.001760]` |

## Checks

- sample count and `signals` / `mu_maps` / `masks` shapes are unchanged.
- `signals`, `mu_maps`, and `masks` values are byte-identical to S208.
- normalized `masks == (mu_maps < 500)` remains true.
- normalized centers fall inside normalized grids.
