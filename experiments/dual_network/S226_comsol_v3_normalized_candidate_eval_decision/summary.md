# S226 normalized V3 candidate evaluation decision

S226 summarizes S222-S225. The coordinate convention issue is fixed, but the current branch candidate is not effective on the normalized V3 hard-case fallback pilot.

## Findings

- S223 zero-shot V2-train to normalized V3 val/test runs without the previous center-bin range error, but val/test IoU is only `0.002348` / `0.012360`.
- S224 normalized V3 train quick probe remains weak. The current candidate reaches train/val/test IoU `0.019538` / `0.047127` / `0.044771`.
- The continuous param-only reference is also weak, with train/val/test IoU `0.039498` / `0.080140` / `0.037464`.
- S225 grouped diagnostics show broad hard-case failure. `bins_correct_center_or_offset_bad` is the most consistent hardest group across zero-shot, candidate, and param-only runs.

## Candidate Status

The S185 `center_bin_offset_plus_grid` candidate remains the current COMSOL parametric candidate for the V2-style branch data, but it is not validated on the normalized V3 hard-case fallback pilot. This is not a main baseline replacement and does not cover true rotated or multi-component COMSOL geometry.

## Next Step

The next single recommended direction is a larger real COMSOL V3 hard-case data pack with true rotated and multi-component geometry coverage. The current fallback pilot is useful as a failure signal, but it is too small and too geometry-limited to justify another model or loss change.
