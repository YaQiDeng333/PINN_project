# S224 normalized V3 train quick probe

S224 tests whether the normalized V3 hard-case train split can be learned by the current candidate, and compares it with the continuous param-only reference. Both runs use normalized V3 train/val/test only.

## Metrics

| run | split | mask_iou | center_grid_mae | x_bin_acc | y_bin_acc | type_acc | axis_mae | rotation_mae | depth_mae |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | train | `0.019538` | `43.068478` | `0.100000` | `0.333333` | `1.000000` | `0.001332` | `0.000142` | `4.964086` |
| candidate | val | `0.047127` | `37.470654` | `0.000000` | `0.200000` | `1.000000` | `0.001388` | `0.000140` | `5.107742` |
| candidate | test | `0.044771` | `44.785847` | `0.100000` | `0.200000` | `1.000000` | `0.001357` | `0.000137` | `4.892258` |
| param_only | train | `0.039498` | `36.937073` | `nan` | `nan` | `1.000000` | `0.001331` | `0.053434` | `4.964127` |
| param_only | val | `0.080140` | `34.018250` | `nan` | `nan` | `1.000000` | `0.001388` | `0.053436` | `5.107620` |
| param_only | test | `0.037464` | `37.423073` | `nan` | `nan` | `1.000000` | `0.001356` | `0.053436` | `4.892380` |

## Interpretation

The normalized V3 train split is not fit well by either configuration. The current center-bin candidate reaches only `0.019538` train IoU, and the param-only reference reaches only `0.039498` train IoU. The candidate is not clearly better than param-only on this pilot: it is lower on train/val IoU and only slightly higher on test IoU.

This indicates that the current V3 fallback pilot is a difficult distribution for the existing parametric candidate, even after coordinate normalization. The result should not be written as a main baseline replacement or as evidence about true rotated/multi-component COMSOL geometry.
