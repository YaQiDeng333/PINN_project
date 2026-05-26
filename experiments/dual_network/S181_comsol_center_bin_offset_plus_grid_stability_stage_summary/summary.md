# S181 COMSOL center-bin offset plus grid stability stage summary

## Decision

S176-S180 made `center_bin_offset_plus_grid` the next candidate to validate, but it did not yet replace the S170 center-grid candidate because S179 was only one recorded seed.

## Evidence

- S178 quick gate selected `center_bin_offset_plus_grid` over the same-seed current candidate reference.
- S179 3000-step confirm reached val / test mask IoU `0.542935` / `0.581320`.
- S179 val / test `center_grid_mae` was `3.362513` / `2.721649`, lower than the S170 center-grid candidate range.

## S181-S185 scope

- Reuse S179 seed1.
- Run only `center_bin_offset_plus_grid` seed2 and seed3.
- Compare against the S170 center-grid candidate historical range.
- Do not rerun `center_bin_offset` alone.
- Do not test new lambda values.
- Do not add raster loss, forward consistency, type/rotation loss, or validation selection.
- Do not modify Python or CLI defaults unless a reproducibility blocker appears.

## Self-review

The next step is a stability validation, not a new route search. Promotion is allowed only if the multi-seed result is stronger than S170 and not dependent on S179 seed1.
