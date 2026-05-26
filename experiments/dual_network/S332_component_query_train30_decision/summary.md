# S332 Component-Query Train30 Decision Summary

S328-S332 did not reach train30. The component-query model and runner smoke passed, but S330 one-sample overfit failed the strict hard-raster gate.

## Decision

- component-query route status: implemented and smoke-tested, but not validated.
- 1-sample gate: failed at IoU `0.974227` against required `>=0.99`.
- 5-sample gate: skipped.
- matched split same-run reference: skipped.
- train30 quick gate: skipped.

## Failure Mode

The failure is not presence/type/bin classification: all were `1.0` on the 1-sample gate. The remaining gap is hard-raster polygon precision: predicted area `194` vs target area `189`, with decoded vertex MAE `5.918177e-06`.

## Next Recommendation

Run a component-query one-sample raster-sensitivity diagnostic before any 5-sample or train30 experiment. The diagnostic should compare vertex/grid error, edge/area error, and pixel disagreement for the failed sample to determine whether this is optimization precision, query-head local shape precision, or hard raster discretization sensitivity.

## Non-Actions

No S185/S181 candidate replacement, no existing runner replacement, no main baseline replacement, no multi-seed, no extra steps, no new COMSOL data, and no push.
