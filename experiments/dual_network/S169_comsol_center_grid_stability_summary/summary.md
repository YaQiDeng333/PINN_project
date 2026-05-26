# S169 COMSOL center-grid stability aggregate summary

## Historical baselines

- Historical param-only val / test IoU: `0.369908` / `0.424462`.
- S161 val / test `center_grid_mae`: `8.017750` / `6.998191`.

## Per-run results

| run_id | seed label | val IoU | test IoU | val center_grid_mae | test center_grid_mae | delta val IoU | delta test IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| existing_unrecorded | existing_unrecorded | 0.469423 | 0.498874 | 5.996350 | 5.546025 | +0.099515 | +0.074412 |
| center_grid_seed1 | 1 | 0.485716 | 0.505590 | 5.443171 | 4.931658 | +0.115808 | +0.081128 |
| center_grid_seed2 | 2 | 0.446966 | 0.503713 | 6.732050 | 4.872537 | +0.077058 | +0.079251 |

## Acceptance criteria

| criterion | threshold / expected | observed | passed |
|---|---|---|---|
| all_val_iou_above_historical_param_only | every run val_iou >= 0.369908 | min=0.446966 | true |
| all_test_iou_above_historical_param_only | every run test_iou >= 0.424462 | min=0.498874 | true |
| median_test_iou_gain_at_least_0.05 | median test_iou >= 0.474462 | median=0.503713 | true |
| at_least_2_of_3_test_iou_ge_0.48 | at least 2/3 test_iou >= 0.480000 | 3/3 | true |
| all_val_center_grid_mae_below_s161 | every run val_center_grid_mae < 8.017750 | max=6.732050 | true |
| all_test_center_grid_mae_below_s161 | every run test_center_grid_mae < 6.998191 | max=5.546025 | true |
| presence_remains_one | all val/test presence_accuracy == 1.0 | all=1.0 | true |
| tradeoffs_do_not_erase_iou_gains | type/axis/rotation tradeoffs do not erase val/test IoU gains | all val/test IoU deltas positive | true |
| improvement_not_existing_unrecorded_only | seed1 and seed2 also improve val/test IoU | seed1=true, seed2=true | true |

## Aggregate judgment

All acceptance criteria passed. The improvement is not dependent on the S164 `existing_unrecorded` run: seed1 and seed2 both improve val/test IoU and reduce center grid error relative to the historical baselines.

## 自评

S169 provides enough evidence to promote `lambda_center_grid=0.1` as the current COMSOL parametric candidate, with the boundary that this is still a branch candidate and not a main baseline replacement.
