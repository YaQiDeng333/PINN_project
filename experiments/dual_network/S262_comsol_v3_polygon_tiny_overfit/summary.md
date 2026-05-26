# S262 COMSOL V3 Polygon Tiny-Overfit Gate

S262 follows the sequential tiny-overfit gate. The model/runner smoke passed, then the one-sample overfit was run on train sample `0`, a true rotated single-component sample.

## One-Sample Result

| metric | value |
| --- | ---: |
| train polygon mask IoU | `0.883178` |
| train polygon Dice | `0.937965` |
| presence accuracy | `1.000000` |
| present type accuracy | `1.000000` |
| normalized vertex MAE | `4.207401e-05` |
| pred area | `214` |
| target area | `189` |

## Gate Decision

The one-sample gate fails because polygon mask IoU is below the stop threshold `0.90`, even though presence/type are correct and vertex MAE is already small.

The 5-sample overfit and train30/val10/test10 quick probe were skipped by design. The failure points to a mismatch between direct vertex regression precision and hard raster IoU tolerance on this grid, not to target/oracle failure: S257 polygon oracle remains `1.000000`.

## Outputs

- `subsets/train_sample0.npz`
- `subsets/train_sample0_polygon_targets.npz`
- `one_sample_overfit/`
- `subsets/train_5sample.npz`
- `subsets/train_5sample_polygon_targets.npz`

The 5-sample subset was prepared before the stop condition was evaluated, but no 5-sample training was run.
