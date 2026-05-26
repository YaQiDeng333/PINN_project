# S280 COMSOL V3 Polygon Train30 Failure Diagnostics

S280 reuses S275 predictions and S276 raster-sensitivity diagnostics. No training is run in this step.

## Outputs

- `train30_failure_table.csv`: per-train-sample IoU, vertex error, FP/FN pixels, area drift, grid-cell vertex displacement, and edge drift.
- `train30_failure_group_summary.csv`: grouped train diagnostics by `hard_case_type`.

## Findings

The failure is broad rather than isolated to one hard-case type. In S275 train diagnostics, `x_bin_wrong_like` is the strongest group but only reaches mean IoU `0.803422`; `geometry_or_type_interaction` and `bins_correct_center_or_offset_bad` are weakest at `0.654147` and `0.682215`.

The worst train sample is sample `21`, `bins_correct_center_or_offset_bad`, with IoU `0.518519`. Its target/pred area is `63` / `60`, so area total is not the main failure. The pixel disagreement is `39` pixels, with false positives / false negatives `18` / `21`. Its max vertex displacement is about `0.853` x-cells and `1.521` y-cells, enough to shift a small hard-rasterized polygon.

There is a systematic over-prediction tendency in S275 train: `25/30` train samples have `pred_area > target_area`, and mean pred/target area is `164.466667` / `143.666667`. However, sample `21` shows that area loss alone is not a sufficient repair because the worst case is a boundary-position error with small area drift.

## Decision

Use the existing runner support first: longer training, then larger capacity only if longer training fails. Do not add area/edge auxiliary unless the simple repair does not pass and diagnostics point to area/edge mismatch as the dominant remaining blocker.
