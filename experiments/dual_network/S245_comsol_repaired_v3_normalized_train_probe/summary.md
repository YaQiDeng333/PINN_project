# S245 repaired COMSOL V3 normalized train quick probe

S245 trains on the normalized repaired V3 train split and evaluates normalized repaired V3 val/test. This stage checks whether the repaired V3 pack is learnable after coordinate normalization.

## Metrics

| run | train IoU | val IoU | test IoU | train center_grid_mae | val center_grid_mae | test center_grid_mae | val x/y bin acc | test x/y bin acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | 1.000000 | 0.055172 | 0.188341 | 0.017676 | 14.200107 | 13.497833 | 0.700000 / 0.100000 | 0.900000 / 0.300000 |
| param_only_reference | 0.773635 | 0.000000 | 0.171178 | 0.746091 | 15.829432 | 12.255308 | n/a | n/a |

## Grouped Diagnostics

S245 groups prediction exports by repaired V3 `hard_case_type` using normalized defect parameters.

- Zero-shot val/test remains near-zero across all hard-case classes. `bins_correct_center_or_offset_bad` is the hardest group in both val and test.
- The normalized V3 candidate fits train perfectly across all hard-case classes.
- Held-out failure is still broad. Candidate val has nonzero IoU only for `both_bins_wrong_like`; candidate test is hardest for `geometry_or_type_interaction` and `rare_y_bin_wrong`.
- The candidate improves test IoU over continuous param-only (`0.188341` vs `0.171178`) but does not solve held-out V3 hard-case generalization.

## Decision

Normalization makes repaired V3 train learnable again, but the `30/10/10` fallback pilot still shows severe held-out split sensitivity. This is not enough evidence to change the current V2-style branch candidate.
