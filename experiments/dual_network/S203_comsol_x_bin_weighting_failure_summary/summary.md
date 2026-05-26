# S203 COMSOL x-bin weighting failure summary

S198-S202 tested the simplest follow-up to S196/S197: increasing pressure on the main center-bin CE, especially x-bin and slots 0/2. The same-round S200 `current_candidate_reference` reproduced the strong S185/S191-level result with val/test IoU `0.546311` / `0.586546`, so the stage did not fail because of a reproduction issue.

The weighted runs did not pass gate:

| run | val IoU | test IoU | val x wrong | test x wrong | val center_grid_mae | test center_grid_mae |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_candidate_reference | 0.546311 | 0.586546 | 0.200000 | 0.133333 | 3.250844 | 2.572883 |
| x_bin_weighted | 0.545284 | 0.555791 | 0.183333 | 0.166667 | 3.420898 | 2.813688 |
| x_bin_slot_weighted | 0.518272 | 0.543191 | 0.250000 | 0.183333 | 5.811699 | 3.065946 |

Decision: freeze the current branch candidate as S185 `center_bin_offset_plus_grid` and stop simple x-bin / y-bin / slot loss-weight tuning. Also stop auxiliary-head, raster, forward-consistency, type/rotation, and dense mask runner detours for this route.

The next stage should use existing exports to package hard samples and design targeted COMSOL hard-case data, not run more training.
