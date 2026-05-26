# S276 COMSOL V3 Polygon Train30 Grouped Diagnostics

S276 groups S275 prediction exports by hard-case type and records worst samples. The train split is the decision split; val/test remain observation only.

## Train Hard-Case Groups

| hard_case_type | samples | mean IoU | min IoU | mean vertex MAE | mean abs area diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| `x_bin_wrong_like` | `10` | `0.803422` | `0.759857` | `1.852272e-04` | `30.0` |
| `both_bins_wrong_like` | `5` | `0.740440` | `0.616667` | `1.728670e-04` | `18.6` |
| `bins_correct_center_or_offset_bad` | `7` | `0.682215` | `0.518519` | `1.573658e-04` | `8.0` |
| `geometry_or_type_interaction` | `5` | `0.654147` | `0.627329` | `1.976818e-04` | `36.2` |
| `rare_y_bin_wrong` | `3` | `0.720231` | `0.678788` | `1.348878e-04` | `14.7` |

## Worst Samples

| split | sample | hard_case_type | IoU | target / pred area | vertex MAE |
| --- | ---: | --- | ---: | --- | ---: |
| train | `21` | `bins_correct_center_or_offset_bad` | `0.518519` | `63 / 60` | `1.676188e-04` |
| val | `1` | `x_bin_wrong_like` | `0.000000` | `210 / 928` | `6.056383e-03` |
| test | `1` | `x_bin_wrong_like` | `0.000000` | `224 / 103` | `1.762167e-03` |

## Interpretation

The failure is broad across train hard-case types. `x_bin_wrong_like` is the best-fitting train group, while `geometry_or_type_interaction` and `bins_correct_center_or_offset_bad` are the weakest. Since train presence/type accuracy is `1.000000`, the next bottleneck is multi-sample vertex precision and shape/raster calibration.

Val/test are very weak, but that is not the stage gate. Because train30 itself fails, the next step should not be multi-seed validation or candidate promotion.
