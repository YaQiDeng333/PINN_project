# S179 COMSOL center-bin offset full confirm

## Executed configuration

S179 ran the S178 winner:

- `center-representation=bin_offset`
- `center-bin-size-cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- `steps=3000`
- `seed=1`

## Results

| split | presence | type_acc | center_grid_mae | x_bin_acc | y_bin_acc | offset_mae | rotation_mae | mask_iou |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1.000000 | 1.000000 | 0.565534 | 1.000000 | 1.000000 | 0.044833 | 0.684530 | 0.716101 |
| val | 1.000000 | 0.633333 | 3.362513 | 0.783333 | 0.883333 | 0.214293 | 8.201941 | 0.542935 |
| test | 1.000000 | 0.650000 | 2.721649 | 0.833333 | 0.950000 | 0.206308 | 7.078501 | 0.581320 |

## Interpretation

The full confirm remains above the S170 current candidate stability runs on val/test IoU and substantially lowers held-out center_grid_mae relative to S170. This is a strong route signal, but it is still a single recorded seed and should not replace the current candidate without a stability repeat.

## Self-review

S179 confirms the center-bin + offset + grid route is worth continuing. It does not yet justify replacing the S170 candidate because S169-level multi-seed stability has not been run.
