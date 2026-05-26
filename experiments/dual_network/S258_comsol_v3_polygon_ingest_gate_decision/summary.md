# S258 COMSOL V3 Polygon Ingest Gate Decision

S254-S257 pass the polygon V3 ingest gate.

## Decision

- The polygon-compatible repaired V3 hard-case pack can be used as the next data source for the polygon inverse route.
- Signal validation passes: finite values, non-near-constant repaired Bz waveforms, complete `x_index`, and expected train/val/test row counts.
- Mask validation passes: `masks == (mu_maps < 500)` mismatch is `0`.
- Polygon target validation passes: embedded polygon arrays and `polygon_params.csv` align for all present components.
- Polygon oracle rasterization passes with train/val/test mean and min IoU `1.000000`.

## Boundaries

- No training was run.
- The existing S185/S181 `center_bin_offset_plus_grid` remains the current V2-style branch candidate.
- This stage is not a main baseline replacement.
- This stage does not return to the old `center + axis + rotation` schema for true V3 geometry.

## Next Step

Proceed to polygon inverse model planning: design a separate polygon inverse runner/head that predicts fixed four-corner vertices, using this ingested polygon V3 pack as the oracle-gated data source.
