# S194 COMSOL signal-to-center auxiliary failure summary

## S191 Quick Gate

S191 compared two auxiliary center-head variants against a same-round 1500-step `current_candidate_reference`.

| run | val IoU | test IoU | val center_grid_mae | test center_grid_mae |
| --- | ---: | ---: | ---: | ---: |
| current_candidate_reference | 0.546311 | 0.586546 | 3.250844 | 2.572883 |
| aux_center_bin_offset | 0.516648 | 0.567790 | 3.958036 | 2.961824 |
| aux_center_bin_offset_xweighted | 0.542723 | 0.580217 | 3.497268 | 2.721402 |

## Decision

The auxiliary head route does not continue in its current form:

- both auxiliary variants had lower val/test IoU than the same-round reference;
- both auxiliary variants worsened held-out `center_grid_mae`;
- S192 full confirm was skipped;
- increasing auxiliary weights would be a blind sweep and is not justified.

The current COMSOL parametric candidate remains the S185 `center_bin_offset_plus_grid` configuration:

- raw MLP;
- shared head;
- fixed-order components;
- `center_representation=bin_offset`;
- `center_bin_size_cells=8`;
- `lambda_center_bin=1.0`;
- `lambda_center_offset=1.0`;
- `lambda_center_grid=0.1`.

## Next Step

S194 moves the route to sample-level center-bin failure diagnostics. The next stage should read existing prediction exports and locate whether the remaining error is driven by x-bin, y-bin, offset, component slot, type, rotation, target area, or a small set of worst validation samples.

No new training is run in S194-S197.
