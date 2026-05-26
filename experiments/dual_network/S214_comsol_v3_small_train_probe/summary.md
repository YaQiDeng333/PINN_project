# S214 COMSOL V3 small-train quick probe

S214 trained on the V3 hard-case train split and evaluated on V3 val/test. This tests whether the V3 fallback pilot is learnable by the current branch candidate and whether the center-bin candidate is better than a continuous param-only reference on the same V3 coordinate system.

## Metrics

| run | split | mask_iou | center_grid_mae | center_x_bin_acc | center_y_bin_acc | center_offset_mae | presence_acc | type_acc | axis_mae | rotation_mae |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_train_candidate | train | 0.019715 | 43.071545 | 0.100000 | 0.333333 | 0.209808 | 1.000000 | 1.000000 | 89.698944 | 0.070656 |
| v3_train_candidate | val | 0.046905 | 37.474541 | 0.000000 | 0.200000 | 0.266497 | 1.000000 | 1.000000 | 92.920029 | 0.070654 |
| v3_train_candidate | test | 0.044968 | 44.786293 | 0.100000 | 0.200000 | 0.212122 | 1.000000 | 1.000000 | 91.538376 | 0.070653 |
| v3_train_param_only_reference | train | 0.038119 | 36.940113 | n/a | n/a | n/a | 1.000000 | 1.000000 | 89.704712 | 0.161146 |
| v3_train_param_only_reference | val | 0.078177 | 34.044182 | n/a | n/a | n/a | 1.000000 | 1.000000 | 92.917526 | 0.161137 |
| v3_train_param_only_reference | test | 0.036448 | 37.399517 | n/a | n/a | n/a | 1.000000 | 1.000000 | 91.548302 | 0.161137 |

## Findings

- The current center-bin candidate does not fit the V3 train split well at 1500 steps: train IoU is only `0.019715`.
- The param-only reference is also weak, with train/val/test IoU `0.038119` / `0.078177` / `0.036448`.
- Presence and type are trivial on this pack because every sample has one rectangular notch; the failure is center/axis geometry, not presence/type.
- Center-bin classification is poor on V3: val/test x-bin accuracy is `0.000000` / `0.100000`, and y-bin accuracy is `0.200000` / `0.200000`.

## Interpretation

The V3 hard-case fallback pack is internally rasterizer-aligned, but the current model configuration does not learn useful V3 geometry from `30` train samples. The result does not justify fine-tuning claims or candidate promotion; it points to a data-coordinate / representation alignment problem that should be resolved before larger V3 model runs.
