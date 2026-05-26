# S206 COMSOL hard-sample / data-design decision

## Decision

Keep the S185 `center_bin_offset_plus_grid` configuration frozen as the current branch COMSOL parametric candidate. It remains branch-local and is not a main baseline replacement.

Stop simple x-bin / y-bin / slot loss-weight tuning. S200 already showed that this route does not improve held-out mask IoU despite a healthy same-round reference.

## Evidence From S204

S204 packaged `23` held-out hard sample keys from existing S200 exports. The hard-sample taxonomy is mixed rather than purely x-bin driven:

- reference hard-sample labels: bins_correct_center_or_offset_bad=5, both_bins_wrong=3, geometry_or_type_interaction=2, x_bin_wrong=12, y_bin_wrong=1;
- low-IoU bins-correct samples: `7`;
- low-IoU x-bin-wrong samples: `15`;
- low-IoU y-bin-wrong samples: `4`.

This means the next route should not be another x-bin CE weight. Some hard samples need better data coverage near x-bin / slot boundaries, while other hard samples require decoded-center / offset / geometry-interaction diagnosis.

## Next Route

Use the S205 hard-case request as the next data-design artifact, and pair it with a bins-correct low-IoU diagnostic before any new training. If future work needs confidence or margin analysis, add a diagnostic export for center-bin logits / softmax margins first; current prediction CSVs cannot support confidence calibration.

## Not Recommended

- Do not replace the S185 candidate.
- Do not continue x-bin / slot-weight sweeps.
- Do not return to aux-head lambda tuning, raster loss, forward consistency, type/rotation loss, or dense conditional mask runner work.
