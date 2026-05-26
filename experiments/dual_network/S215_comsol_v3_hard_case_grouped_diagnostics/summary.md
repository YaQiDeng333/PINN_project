# S215 V3 hard-case grouped diagnostics

S215 groups the S214 prediction exports by `hard_case_type`. S213 has no prediction export because zero-shot evaluation was blocked by a V2/V3 grid-coordinate mismatch.

## Inputs

- `v3_train_candidate`: S214 V3 train candidate, `center_representation=bin_offset`
- `v3_train_param_only_reference`: S214 V3 train continuous param-only reference
- defect metadata: `experiments/dual_network/S208_comsol_v3_hard_case_ingest/raw/*/defect_params.csv`
- converted grids: `experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted/*_comsol_v3_hard_case.npz`

## Grouped Findings

| run | split | hardest hard_case_type | IoU | center_grid_mae | x_bin_acc | y_bin_acc |
|---|---|---|---:|---:|---:|---:|
| v3_train_candidate | val | bins_correct_center_or_offset_bad | 0.000000 | 50.118175 | 0.000000 | 0.000000 |
| v3_train_candidate | test | bins_correct_center_or_offset_bad | 0.000000 | 66.334725 | 0.000000 | 0.000000 |
| v3_train_param_only_reference | val | bins_correct_center_or_offset_bad | 0.000000 | 48.868240 | 0.000000 | 0.000000 |
| v3_train_param_only_reference | test | bins_correct_center_or_offset_bad | 0.000000 | 39.232221 | 0.000000 | 0.500000 |

## Required Answers

1. Hardest hard_case type: no single isolated class explains the failure. `bins_correct_center_or_offset_bad` is consistently among the worst on held-out splits, but `both_bins_wrong_like`, `rare_y_bin_wrong`, and `geometry_or_type_interaction` also have near-zero IoU in several groups.
2. Zero-shot failure concentration: zero-shot produced no valid prediction export because V2 and V3 center-bin grids use incompatible coordinate ranges.
3. V3 train improvement by type: V3 train did not meaningfully improve any class. The center-bin candidate reaches val/test IoU `0.046905` / `0.044968`; param-only reaches `0.078177` / `0.036448`.
4. Data sufficiency: the pack is sufficient to reveal a major coordinate / representation issue, but not sufficient to guide a final V3 route because it contains only single rectangular Block solved geometry.
5. Larger V3 pack: do not scale this exact setup yet. First fix or standardize V3 geometry units / coordinates relative to the V2 parametric route, then regenerate or reconvert before larger rotated / multi-component packs.

## Notes

`center_offset_mae` in `grouped_by_hard_case_type.csv` is derived from decoded centers and bin-normalized residuals because raw offset logits are not exported.
