# S282 COMSOL V3 Polygon Train30 Repair Quick Gate

S282 runs the train30 repair gate with stop-on-pass behavior. The S275 run is reused as `current_train30_reference`.

## Matrix

| run | status | train mean IoU | train min IoU | train vertex MAE | train presence/type | val IoU obs | test IoU obs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_train30_reference | reused S275 | `0.731445` | `0.518519` | `1.793932e-04` | `1.000000` / `1.000000` | `0.033122` | `0.089484` |
| longer_train30 | passed | `0.935101` | `0.802920` | `5.560893e-05` | `1.000000` / `1.000000` | `0.046352` | `0.136720` |
| larger_capacity_train30 | skipped | - | - | - | - | - | - |
| area_edge_aux_train30 | skipped | - | - | - | - | - | - |

`longer_train30` uses the S275 configuration with only `steps` increased from `10000` to `20000`.

## Hard-Case Group Gate

Train grouped IoU for `longer_train30`:

| hard_case_type | mean IoU | min IoU |
| --- | ---: | ---: |
| x_bin_wrong_like | `0.968752` | `0.931034` |
| both_bins_wrong_like | `0.968497` | `0.952381` |
| bins_correct_center_or_offset_bad | `0.934395` | `0.875000` |
| geometry_or_type_interaction | `0.868425` | `0.816176` |
| rare_y_bin_wrong | `0.880043` | `0.802920` |

All train hard-case groups clear the `0.75` mean IoU floor.

## Decision

The train30 fit gate passes with the minimal repair, `longer_train30`. By stop-on-pass, `larger_capacity_train30` and `area_edge_aux_train30` are not run.

Val/test remain observation-only and weak; they are not used to pass or fail this stage.
