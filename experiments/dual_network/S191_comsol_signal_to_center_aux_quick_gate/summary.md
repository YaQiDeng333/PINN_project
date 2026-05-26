# S191 COMSOL signal-to-center auxiliary quick gate

## Setup

All runs used the same 1500-step seed-1 setup:

- raw MLP;
- shared head;
- fixed-order components;
- `center_representation=bin_offset`;
- `center_bin_size_cells=8`;
- `lambda_center_bin=1.0`;
- `lambda_center_offset=1.0`;
- `lambda_center_grid=0.1`;
- no raster loss;
- no forward consistency;
- no validation-aware endpoint selection.

The same-round reference was the primary baseline. Historical S185 results were used only as sanity context.

## Results

| run | val IoU | test IoU | val center_grid_mae | test center_grid_mae | val x-bin acc | test x-bin acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_candidate_reference | 0.546311 | 0.586546 | 3.250844 | 2.572883 | 0.783333 | 0.850000 |
| aux_center_bin_offset | 0.516648 | 0.567790 | 3.958036 | 2.961824 | 0.750000 | 0.850000 |
| aux_center_bin_offset_xweighted | 0.542723 | 0.580217 | 3.497268 | 2.721402 | 0.800000 | 0.866667 |

## Gate Decision

No auxiliary group passed the quick gate.

- `aux_center_bin_offset` reduced train center error but lowered both val/test IoU versus the same-round reference.
- `aux_center_bin_offset_xweighted` improved train fitting and slightly improved x-bin accuracy, but still lowered val/test IoU versus the same-round reference and worsened held-out `center_grid_mae`.
- Presence stayed at `1.0`, so the failure is not a presence collapse.

Because the same-round reference reproduced a strong S178/S185-level result, this is not treated as a reproduce-risk failure. The conclusion is that the auxiliary head did not improve held-out center-bin stability in this quick gate.

## Self-Review

- The decision uses the same-round `current_candidate_reference`, not only S185 historical metrics.
- S192 should be skipped because no aux group satisfies val/test IoU and center-grid criteria.
- The current candidate remains S185 `center_bin_offset_plus_grid`.
