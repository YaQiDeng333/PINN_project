# S223 normalized V3 zero-shot evaluation

S223 reran the V2-train to normalized-V3 val/test evaluation because S220 was only a 5-step runability gate. The command used the current S185 center-bin candidate configuration and exported predictions.

## Metrics

| split | mask_iou | center_grid_mae | x_bin_acc | y_bin_acc | type_acc | axis_mae | rotation_mae | depth_mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | `0.002348` | `58.098274` | `0.000000` | `0.000000` | `0.000000` | `0.006175` | `115.285599` | `78.498245` |
| test | `0.012360` | `61.760284` | `0.000000` | `0.100000` | `0.000000` | `0.006088` | `115.285599` | `76.498245` |

## Interpretation

The normalized coordinate convention is now runnable, but V2-trained zero-shot generalization to this V3 hard-case pilot is not effective. The held-out IoU is far below the V2 held-out candidate range, and center-bin accuracy is essentially absent on normalized V3 val/test.

The run also exposes a target-distribution difference: the V2 training target vocabulary contains `rectangular_notch, rotated_rect`, while normalized V3 contains only `rectangular_notch`. The single-type V3 simplification does not rescue zero-shot performance; type accuracy is `0.0` on val/test.

The grouped S225 diagnostics should be used for hard-case localization. This summary does not claim anything about main baseline replacement.
