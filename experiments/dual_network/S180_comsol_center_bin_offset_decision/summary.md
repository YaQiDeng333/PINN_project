# S180 COMSOL center-bin offset decision

## Decision

Continue the center-bin + offset route. Do not yet replace the current COMSOL parametric candidate.

## Evidence

- S178 quick gate selected `center_bin_offset_plus_grid`.
- S179 3000-step confirm produced val / test mask IoU `0.542935` / `0.581320`.
- The S179 val/test center_grid_mae was `3.362513` / `2.721649`, lower than the S170 center-grid candidate runs.
- Presence remained `1.0`.

## Current candidate status

The current branch candidate remains:

`raw MLP / shared head / fixed-order + lambda_center_grid=0.1`

The new `center_bin_offset_plus_grid` configuration becomes the next candidate to validate with S169-style stability repeat. It should not be promoted until multi-seed validation confirms that the gain is not seed-specific.

## Next step

Run a stability stage for `center_bin_offset_plus_grid`:

- reuse S179 as `seed1`
- add at least seed2 / seed3
- compare against the S170 center-grid candidate band
- promote only if val/test IoU and center_grid_mae remain stable across seeds

## Stop condition

If stability repeat shows mixed or seed-dependent gains, stop this route and move to `signal-to-center auxiliary head` rather than tuning center-bin or center-grid lambdas.

## Self-review

S180 keeps the route evidence bounded: center-bin + offset is promising, but not yet the default branch candidate.
