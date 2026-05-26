# S272 COMSOL V3 Polygon 5-Sample Diagnostics

S272 diagnoses the passed 5-sample overfit result by sample and hard-case label.

## Per-Sample Metrics

| subset sample | source sample | hard_case_type | IoU | vertex MAE | pred / target area |
| ---: | ---: | --- | ---: | ---: | --- |
| `0` | `0` | `x_bin_wrong_like` | `0.994737` | `4.186586e-06` | `190 / 189` |
| `1` | `11` | `both_bins_wrong_like` | `1.000000` | `5.049249e-06` | `184 / 184` |
| `2` | `15` | `bins_correct_center_or_offset_bad` | `1.000000` | `7.747069e-06` | `76 / 76` |
| `3` | `22` | `geometry_or_type_interaction` | `1.000000` | `5.827646e-06` | `102 / 102` |
| `4` | `27` | `rare_y_bin_wrong` | `0.985401` | `3.828958e-06` | `135 / 137` |

## Findings

- Every hard-case type is represented by one sample.
- Both multi-component samples fit with IoU `1.000000`.
- The worst sample is `rare_y_bin_wrong` source sample `27`, but it still passes with IoU `0.985401` and area diff `-2` pixels.
- Raster sensitivity is now small: maximum component-level vertex displacement is about `0.020669` x-cells and `0.067136` y-cells.

## Outputs

- `per_sample_5sample_diagnostics.csv`
- `grouped_by_hard_case_type.csv`
- `worst_5sample_polygon_samples.csv`
- `raster_sensitivity/`
