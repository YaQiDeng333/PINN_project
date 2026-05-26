# S215 V3 hard-case grouped diagnostics

This summary is generated from existing prediction exports and defect parameters only.

## Runs

- `normalized_v3_candidate`: splits=test,train,val, mean IoU=0.648703, mean center_grid_mae=5.550193
- `normalized_v3_param_only`: splits=test,train,val, mean IoU=0.498417, mean center_grid_mae=6.064602

## Hardest groups

- `normalized_v3_candidate` `test` hardest: `rare_y_bin_wrong` (count=1, IoU=0.000000, center_grid_mae=31.867894, x_bin_acc=1.000000, y_bin_acc=0.000000)
- `normalized_v3_candidate` `train` hardest: `bins_correct_center_or_offset_bad` (count=7, IoU=1.000000, center_grid_mae=0.021235, x_bin_acc=1.000000, y_bin_acc=1.000000)
- `normalized_v3_candidate` `val` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=13.869036, x_bin_acc=1.000000, y_bin_acc=0.000000)
- `normalized_v3_param_only` `test` hardest: `rare_y_bin_wrong` (count=1, IoU=0.000000, center_grid_mae=30.896573, x_bin_acc=1.000000, y_bin_acc=0.000000)
- `normalized_v3_param_only` `train` hardest: `geometry_or_type_interaction` (count=5, IoU=0.605728, center_grid_mae=0.742790, x_bin_acc=1.000000, y_bin_acc=0.800000)
- `normalized_v3_param_only` `val` hardest: `bins_correct_center_or_offset_bad` (count=2, IoU=0.000000, center_grid_mae=16.005141, x_bin_acc=0.500000, y_bin_acc=0.000000)

## Interpretation

- The script reports grouped evidence; the stage-level conclusion should compare these tables with S213/S214 summaries.
- `center_offset_mae` is derived from decoded center coordinates and bin-normalized residuals because raw offset logits are not exported.
