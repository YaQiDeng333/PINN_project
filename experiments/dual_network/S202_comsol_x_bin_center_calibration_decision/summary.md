# S202 COMSOL x-bin center calibration decision

## Decision

The x-bin weighting / slot-aware weighting route does not continue in its current simple form.

The current COMSOL parametric candidate remains S185 `center_bin_offset_plus_grid`:

- raw MLP;
- shared head;
- fixed-order components;
- `center_representation=bin_offset`;
- `center_bin_size_cells=8`;
- `lambda_center_bin=1.0`;
- `lambda_center_offset=1.0`;
- `lambda_center_grid=0.1`;
- no auxiliary head.

## Interpretation

S200 shows that directly increasing main x-bin CE pressure is not enough. The x-only weighted run slightly reduced val x wrong rate, but this did not improve mask IoU and made test x-bin behavior worse. Slot-aware weighting overfit train and harmed held-out center localization.

This suggests the residual errors are not solved by a simple CE reweighting knob. The next step should avoid further x-bin lambda/slot-weight sweeps.

## Recommended Next Direction

Keep the S185 candidate unchanged. The next route should be diagnostic rather than another weighting sweep:

- inspect low-IoU samples where bins are correct but offset / decoded center / geometry interaction still fails;
- compare center-bin confidence and distance-to-bin-boundary if logits are exported in a future diagnostic;
- consider representation calibration only if a non-loss-weighting failure mode is identified.

## Boundary

S201 did not run. No candidate replacement is made. Any future replacement still requires a separate multi-seed validation stage.
