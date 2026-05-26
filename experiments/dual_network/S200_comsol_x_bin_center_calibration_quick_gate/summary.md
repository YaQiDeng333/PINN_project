# S200 COMSOL x-bin center calibration quick gate

## Setup

S200 ran a same-round 1500-step seed-1 quick gate. All training commands explicitly used `--export-predictions`, then `comsol_center_bin_failure_diagnostics.py` was run on each output directory.

Runs:

- `current_candidate_reference`: S185 candidate unchanged.
- `x_bin_weighted`: `center_bin_x_weight=1.5`, `center_bin_y_weight=1.0`.
- `x_bin_slot_weighted`: same x/y weights plus `center_bin_slot_weights=1.5,1.0,1.5`.

## Results

| run | val IoU | test IoU | val x wrong | test x wrong | val y wrong | test y wrong | val center_grid_mae | test center_grid_mae |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_candidate_reference | 0.546311 | 0.586546 | 0.200000 | 0.133333 | 0.083333 | 0.033333 | 3.250844 | 2.572883 |
| x_bin_weighted | 0.545284 | 0.555791 | 0.183333 | 0.166667 | 0.150000 | 0.033333 | 3.420898 | 2.813688 |
| x_bin_slot_weighted | 0.518272 | 0.543191 | 0.250000 | 0.183333 | 0.216667 | 0.083333 | 5.811699 | 3.065946 |

## Gate Decision

No x-bin calibration run passed the quick gate.

- `x_bin_weighted` slightly reduced val x wrong rate, but val IoU did not improve, test IoU dropped, test x wrong rate increased, and held-out center-grid error worsened.
- `x_bin_slot_weighted` improved train fitting but substantially degraded val/test IoU, x/y wrong rates, and center-grid error.
- Presence stayed at `1.0`, so the failure is not a presence collapse.

Because the same-round reference reproduced the S191 value exactly, this is not a reproduction-risk failure. The route failed because x-bin weighting did not translate into held-out mask improvement.

## Self-Review

- S200 judged candidates against the same-round current-candidate reference.
- Prediction exports were present for every run.
- X wrong rate alone was not treated as sufficient for promotion.
