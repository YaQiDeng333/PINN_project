# Center-Anchored Polygon Y-Bin Diagnostics

This diagnostic reads existing S300 matched-coverage predictions only; it does not run training.

## Split Summary

| split | samples | zero_iou | x_bin_acc | y_bin_acc | y_within1 | mean_abs_y_error | mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `30` | `0` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.991753` |
| val | `10` | `9` | `0.230769` | `0.153846` | `0.538462` | `1.692308` | `0.017919` |
| test | `10` | `9` | `0.666667` | `0.083333` | `0.333333` | `2.000000` | `0.068354` |

## Findings

- heldout zero-IoU present components: `23`
- zero-IoU components with y-bin error: `22`
- zero-IoU components with x-bin error: `13`
- heldout y-bin adjacent errors: `8`
- heldout y-bin distance >=2 errors: `14`

Y-bin error is reported as ordered distance, so adjacent-bin and far-bin failures are separated instead of collapsed into ordinary classification errors.
