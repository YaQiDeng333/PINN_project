# S196 COMSOL center-bin failure diagnostics

## Scope

S196 ran `comsol_center_bin_failure_diagnostics.py` on the three existing S191 prediction export directories:

- `current_candidate_reference`;
- `aux_center_bin_offset`;
- `aux_center_bin_offset_xweighted`.

No new training was run.

## Aggregate Results

| label | split | IoU | center_grid_error | x wrong rate | y wrong rate | both bins correct | IoU<0.50 samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_candidate_reference | val | 0.546311 | 3.250845 | 0.200000 | 0.083333 | 0.733333 | 7 |
| current_candidate_reference | test | 0.586546 | 2.572883 | 0.133333 | 0.033333 | 0.833333 | 2 |
| aux_center_bin_offset | val | 0.516648 | 3.958036 | 0.233333 | 0.133333 | 0.650000 | 9 |
| aux_center_bin_offset | test | 0.567790 | 2.961824 | 0.150000 | 0.150000 | 0.716667 | 5 |
| aux_center_bin_offset_xweighted | val | 0.542723 | 3.497268 | 0.183333 | 0.166667 | 0.683333 | 7 |
| aux_center_bin_offset_xweighted | test | 0.580217 | 2.721402 | 0.150000 | 0.133333 | 0.733333 | 5 |

## Findings

1. x-bin errors have the larger impact than y-bin errors in the current candidate. For `current_candidate_reference`, val x/y wrong rates are `0.200000` / `0.083333`, and test x/y wrong rates are `0.133333` / `0.033333`.

2. Val fluctuation is partly concentrated in worst samples, but not exclusively. The current candidate has 7 val samples below IoU `0.50`; some worst samples still have both bins correct, which means decoded center/offset magnitude and geometry interaction still matter after bin correctness.

3. Component slots 0 and 2 are more error-prone than slot 1 in the current candidate. On val, slot 0 and slot 2 have x wrong rate `0.30`, while slot 1 has x wrong rate `0.00`.

4. Target area matters. In the current candidate grouped output, `1000-1500` area samples have higher x wrong rate than `500-1000` area samples, and the lowest IoU bin has much larger center error.

5. The auxiliary head did not improve because it did not improve the final decoded center-bin behavior used for rasterization. The plain aux variant worsened both x and y wrong rates. The x-weighted aux variant slightly reduced val x wrong rate, but y wrong rate increased and held-out center_grid_error worsened, so mask IoU still declined versus the same-round reference.

## Next Step

Do not continue the current auxiliary-head route. The next route should focus on x-bin-centered refinement of the main center-bin output, with hard-sample / bin-error calibration as the first diagnostic target rather than another auxiliary head or lambda sweep.
