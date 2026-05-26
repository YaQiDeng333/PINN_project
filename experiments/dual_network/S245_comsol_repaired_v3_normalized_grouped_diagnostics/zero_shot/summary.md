# S215 V3 hard-case grouped diagnostics

This summary is generated from existing prediction exports and defect parameters only.

## Runs

- `zero_shot_v2_to_repaired_v3_normalized`: splits=test,val, mean IoU=0.006432, mean center_grid_mae=54.837077

## Hardest groups

- `zero_shot_v2_to_repaired_v3_normalized` `test` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=71.273342, x_bin_acc=0.000000, y_bin_acc=0.000000)
- `zero_shot_v2_to_repaired_v3_normalized` `val` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=65.703913, x_bin_acc=0.000000, y_bin_acc=0.000000)

## Interpretation

- The script reports grouped evidence; the stage-level conclusion should compare these tables with S213/S214 summaries.
- `center_offset_mae` is derived from decoded center coordinates and bin-normalized residuals because raw offset logits are not exported.
