# S225 normalized V3 hard-case grouped diagnostics

This summary is generated from S223/S224 prediction exports, normalized V3 defect parameters, and normalized V3 grids. It uses val/test splits only, because the S223 train predictions come from V2 train and must not be joined against V3 train hard-case labels.

## Runs

- `v3_train_candidate`: splits=test,val, mean IoU=0.045949, mean center_grid_mae=41.128251
- `v3_train_param_only`: splits=test,val, mean IoU=0.058802, mean center_grid_mae=35.720662
- `zero_shot_v2_train`: splits=test,val, mean IoU=0.007354, mean center_grid_mae=59.929280

## Hardest groups

- `v3_train_candidate` `test` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=66.328547, x_bin_acc=0.000000, y_bin_acc=0.000000)
- `v3_train_candidate` `val` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=50.112497, x_bin_acc=0.000000, y_bin_acc=0.000000)
- `v3_train_param_only` `test` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=39.366313, x_bin_acc=0.000000, y_bin_acc=0.500000)
- `v3_train_param_only` `val` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=48.853917, x_bin_acc=0.000000, y_bin_acc=0.000000)
- `zero_shot_v2_train` `test` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=91.318417, x_bin_acc=0.000000, y_bin_acc=0.000000)
- `zero_shot_v2_train` `val` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=67.736256, x_bin_acc=0.000000, y_bin_acc=0.000000)

## Interpretation

- `bins_correct_center_or_offset_bad` is the most consistent hardest group. It is the hardest val/test group for zero-shot, V3-train candidate, and V3-train param-only runs.
- Zero-shot failure is broad rather than confined to one hard-case type. `x_bin_wrong_like`, `both_bins_wrong_like`, and `bins_correct_center_or_offset_bad` all have near-zero or very low IoU.
- V3 train improves some classes relative to zero-shot, especially `geometry_or_type_interaction` on candidate val and `x_bin_wrong_like` on candidate/param-only test, but it does not fix the systematic bin/center failure.
- The current pilot is sufficient to show that normalized V3 remains a hard distribution, but it is not sufficient to drive the next training direction by itself. It is single rectangular Block data and does not cover true rotated/multi-component geometry.
- `center_offset_mae` is derived from decoded center coordinates and bin-normalized residuals because raw offset logits are not exported.

## Answers

1. Hardest class: `bins_correct_center_or_offset_bad` is the most stable hardest group.
2. Zero-shot failure concentration: failure is broad, with `x_bin_wrong_like`, `both_bins_wrong_like`, and `bins_correct_center_or_offset_bad` all weak.
3. V3 train improvement: V3 training helps some grouped IoU values but does not produce a usable candidate-level result.
4. Current pack sufficiency: the pilot is useful for exposing failure, but too small and too geometry-limited for the next training decision.
5. Larger V3 pack: yes, the next data step should expand real V3 hard-case coverage, especially true rotated and multi-component geometry.
