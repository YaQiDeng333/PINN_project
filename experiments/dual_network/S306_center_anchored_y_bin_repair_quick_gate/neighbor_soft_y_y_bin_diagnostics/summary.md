# Center-Anchored Polygon Y-Bin Diagnostics

This diagnostic reads existing S300 matched-coverage predictions only; it does not run training.

## Split Summary

| split | samples | zero_iou | x_bin_acc | y_bin_acc | y_within1 | mean_abs_y_error | mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `30` | `0` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.988315` |
| val | `10` | `7` | `0.307692` | `0.307692` | `0.692308` | `1.384615` | `0.056407` |
| test | `10` | `8` | `0.833333` | `0.166667` | `0.500000` | `1.666667` | `0.069911` |

## Findings

- heldout zero-IoU present components: `20`
- zero-IoU components with y-bin error: `18`
- zero-IoU components with x-bin error: `10`
- heldout y-bin adjacent errors: `9`
- heldout y-bin distance >=2 errors: `10`

Y-bin error is reported as ordered distance, so adjacent-bin and far-bin failures are separated instead of collapsed into ordinary classification errors.
