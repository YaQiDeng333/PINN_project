# S240 repaired V3 hard-case grouped diagnostics

This summary is generated from S239 prediction exports, repaired V3 `defect_params.csv`, and converted repaired V3 grids. S238 zero-shot has no prediction exports because it failed before evaluation on a cross-grid center-bin range check.

## Runs

- `repaired_v3_train_candidate`: splits=`train,val,test`, mean sample IoU=`0.649314`, mean center_grid_mae=`5.561871`.
- `repaired_v3_train_param_only_reference`: splits=`train,val,test`, mean sample IoU=`0.623726`, mean center_grid_mae=`5.811345`.

## Hardest Groups

- candidate val: `bins_correct_center_or_offset_bad`, `geometry_or_type_interaction`, `rare_y_bin_wrong`, and `x_bin_wrong_like` all have mean IoU `0.000000`; only `both_bins_wrong_like` is nonzero at `0.264368`.
- candidate test: `geometry_or_type_interaction` and `rare_y_bin_wrong` are hardest with IoU `0.000000`; `both_bins_wrong_like` is `0.250000`, `x_bin_wrong_like` is `0.257701`, and `bins_correct_center_or_offset_bad` is `0.349162`.
- candidate train: all groups are near-perfect, from `0.993103` to `1.000000`, so the repaired V3 signal can support train fitting.

## Answers

1. Hardest hard-case types: held-out failures are broad. Val is almost entirely failed except `both_bins_wrong_like`; test is weakest for `geometry_or_type_interaction` and `rare_y_bin_wrong`.
2. Zero-shot grouped failure cannot be assessed because S238 did not reach prediction export.
3. Repaired V3 train improves train fitting dramatically, but held-out val/test remain weak.
4. The current fallback pack is useful as a repaired-signal learnability gate, but too small and split-sensitive to guide a stable candidate decision.
5. Next data action should be a larger repaired V3 hard-case pack before mixed training or multi-seed candidate validation.

`center_offset_mae` is derived from decoded center coordinates and bin-normalized residuals because raw offset logits are not exported.
