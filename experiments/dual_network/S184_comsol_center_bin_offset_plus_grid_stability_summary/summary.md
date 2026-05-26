# S184 COMSOL center-bin offset plus grid stability aggregate

## Comparison table

| run | seed | source | val IoU | test IoU | val center_grid_mae | test center_grid_mae | val x/y bin acc | test x/y bin acc | val/test presence | val/test type acc | val/test axis MAE | val/test rotation MAE |
|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|
| S170 center-grid range | range | S169/S170 | 0.446966-0.485716 | 0.498874-0.505590 | worst 6.732050 | worst 5.546025 | NA | NA | 1.0 / 1.0 | NA | NA | NA |
| center_bin_offset_plus_grid_seed1 | 1 | S179 | 0.542935 | 0.581320 | 3.362513 | 2.721649 | 0.783333 / 0.883333 | 0.833333 / 0.950000 | 1.0 / 1.0 | 0.633333 / 0.650000 | 0.000594 / 0.000502 | 8.201941 / 7.078501 |
| center_bin_offset_plus_grid_seed2 | 2 | S183 | 0.484303 | 0.575504 | 6.282760 | 2.929023 | 0.716667 / 0.800000 | 0.833333 / 0.916667 | 1.0 / 1.0 | 0.616667 / 0.583333 | 0.000585 / 0.000568 | 8.346560 / 7.979585 |
| center_bin_offset_plus_grid_seed3 | 3 | S183 | 0.492127 | 0.578738 | 6.026593 | 2.804331 | 0.800000 / 0.833333 | 0.850000 / 0.900000 | 1.0 / 1.0 | 0.650000 / 0.666667 | 0.000593 / 0.000583 | 8.074749 / 8.185495 |

## Acceptance criteria

| criterion | threshold / expected | observed | passed |
|---|---|---|---|
| all_val_iou_above_s170_min | every run val_iou > 0.446966 | min=0.484303 | true |
| all_test_iou_above_s170_min | every run test_iou > 0.498874 | min=0.575504 | true |
| median_test_iou_ge_0_55 | median test_iou >= 0.55 | median=0.578738 | true |
| at_least_2_of_3_test_iou_ge_0_56 | at least 2/3 test_iou >= 0.56 | 3/3 | true |
| all_val_center_grid_mae_below_s170_worst | every run val_center_grid_mae < 6.732050 | max=6.282760 | true |
| all_test_center_grid_mae_below_s170_worst | every run test_center_grid_mae < 5.546025 | max=2.929023 | true |
| held_out_bin_accuracy_not_collapsed | val/test x/y bin accuracy about >= 0.70 | min=0.716667 | true |
| presence_remains_one | all val/test presence_accuracy == 1.0 | all=1.0 | true |
| tradeoffs_do_not_erase_iou_gains | type/axis/rotation tradeoffs do not erase mask IoU gains | all val/test IoU above S170 lower bound | true |
| improvement_not_seed1_only | seed2 or seed3 remains close enough to S179 improvement | seed2/test=0.575504, seed3/test=0.578738 | true |

## Aggregate judgment

All acceptance criteria passed. Test IoU is stable across three runs (`0.575504-0.581320`) and clearly exceeds the S170 center-grid range (`0.498874-0.505590`). Val IoU remains above the S170 lower bound for all runs, but seed2/seed3 are materially lower than S179 seed1. The configuration is strong enough to promote as the current branch candidate, with an explicit caveat that the next center-bin stage should continue monitoring val stability.

## Self-review

S184 compares S170 and S181-S185 in one table and does not hide per-seed variation behind averages. The result supports promotion, but not a final-solution claim.
