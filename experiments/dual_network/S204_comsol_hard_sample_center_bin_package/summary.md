# S204 COMSOL hard-sample center-bin package

## Inputs

This package uses only existing S200 prediction exports and S200/S196-style center-bin diagnostics. No new training, no new seed, and no runner change was used.

Runs included:

- `current_candidate_reference`;
- `x_bin_weighted`;
- `x_bin_slot_weighted`.

## Hard-Sample Definition

Hard samples are held-out val/test samples from `current_candidate_reference` with `mask_iou < 0.60`, plus the lowest five val/test samples for coverage. This produced `23` unique held-out sample keys.

## Failure Taxonomy

Reference hard-sample labels: bins_correct_center_or_offset_bad=5, both_bins_wrong=3, geometry_or_type_interaction=2, x_bin_wrong=12, y_bin_wrong=1.

Held-out low-IoU samples below `0.60`: bins_correct_center_or_offset_bad=5, both_bins_wrong=3, geometry_or_type_interaction=2, x_bin_wrong=12, y_bin_wrong=1.

- x-bin wrong low-IoU samples: `15`.
- y-bin wrong low-IoU samples: `4`.
- bins-correct low-IoU samples: `7`.

The bins-correct low-IoU group matters because it shows the residual failure is not fully explained by bin classification. Offset magnitude, decoded center, area, type sequence, and geometry interaction still need diagnosis.

## Worst Current-Candidate Held-Out Samples

| split | sample_index | mask_iou | failure_label | mean_center_grid_error | max_center_grid_error | any_x_wrong | any_y_wrong |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| val | 12 | 0.385101 | both_bins_wrong | 6.414969 | 9.467140 | true | true |
| val | 5 | 0.418361 | x_bin_wrong | 5.166166 | 6.788209 | true | false |
| val | 0 | 0.435714 | x_bin_wrong | 4.507149 | 7.554158 | true | false |
| val | 17 | 0.447869 | both_bins_wrong | 4.847379 | 5.793482 | true | true |
| val | 11 | 0.454741 | x_bin_wrong | 4.521057 | 4.704295 | true | false |
| test | 5 | 0.467123 | x_bin_wrong | 4.156143 | 4.229502 | true | false |
| test | 4 | 0.487500 | both_bins_wrong | 4.431269 | 5.821475 | true | true |
| val | 18 | 0.493298 | y_bin_wrong | 5.752344 | 8.026397 | false | true |
| val | 3 | 0.497720 | x_bin_wrong | 3.727690 | 5.029976 | true | false |
| test | 11 | 0.504449 | x_bin_wrong | 4.179949 | 6.328369 | true | false |

## Delta Interpretation

`hard_sample_summary.csv` contains one row per current-candidate hard sample, while `hard_sample_run_comparison.csv` contains the same hard sample keys across all S200 runs. `run_delta_summary.csv` compares `x_bin_weighted` and `x_bin_slot_weighted` against `current_candidate_reference` by `split + sample_index`. The S200 result shows direct x-bin CE weighting did not solve the failure mode: `x_bin_weighted` slightly reduced some val x-bin errors but worsened test IoU and center-grid error, while slot-aware weighting degraded held-out behavior more broadly.

## Known Missing Signal

The existing CSVs do not contain raw center-bin logits, softmax confidence, or bin margins. This package can rank and categorize hard samples, but it cannot perform confidence calibration. A future confidence diagnostic would need a new export, not a new training run.

## Next Use

Use this package to decide whether the next stage should request targeted COMSOL hard cases or add a diagnostic export for bins-correct low-IoU samples. Do not use it to justify another x-bin / slot-weight sweep.
