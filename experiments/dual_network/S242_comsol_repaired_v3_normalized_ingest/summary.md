# S242 repaired COMSOL V3 normalized ingest

S242 creates a normalized copy of the repaired V3 hard-case pack. Source data under S233 and the raw COMSOL export root are not modified.

## Transform

- `x_norm = (x_raw - 2250.0) * (0.08 / 4500.0)`
- `y_norm = (y_raw - 1500.0) * (0.02 / 3000.0)`
- `defect_center_x/y` use the same affine transform.
- `defect_axis_x *= 0.08 / 4500.0`
- `defect_axis_y *= 0.02 / 3000.0`
- `signals`, `mu_maps`, and `masks` are unchanged.

## Ranges

| split | shape | raw x | normalized x | raw y | normalized y | raw center x | normalized center x | raw center y | normalized center y | raw axis x | normalized axis x | raw axis y | normalized axis y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | `(30, 3, 200)` | `0..4500` | `-0.04..0.04` | `0..3000` | `-0.01..0.01` | `704.07..3770.4` | `-0.0274832..0.0270294` | `810..2342.42` | `-0.0046..0.00561616` | `420..880` | `0.00746667..0.0156444` | `110..260` | `0.000733333..0.00173333` |
| val | `(10, 3, 200)` | `0..4500` | `-0.04..0.04` | `0..3000` | `-0.01..0.01` | `781.407..3359.69` | `-0.0261083..0.0197277` | `810..1856.06` | `-0.0046..0.00237374` | `450..880` | `0.008..0.0156444` | `120..260` | `0.0008..0.00173333` |
| test | `(10, 3, 200)` | `0..4500` | `-0.04..0.04` | `0..3000` | `-0.01..0.01` | `1125.52..3770.4` | `-0.0199908..0.0270294` | `1200..2298.86` | `-0.002..0.00532576` | `420..840` | `0.00746667..0.0149333` | `110..260` | `0.000733333..0.00173333` |

## Depth / z

Depth and z values are retained in raw COMSOL units. The current hard rasterizer does not use depth/z to generate the 2D mask, and the repaired V3 z convention has not been audited enough to introduce a reliable affine mapping.
