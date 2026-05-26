# S268 COMSOL V3 Polygon One-Sample Repair Decision

S264-S268 repairs the immediate polygon inverse one-sample gate without entering 5-sample or train30.

## Result

The one-sample gate now passes with `longer_overfit`:

- hard polygon mask IoU: `1.000000`
- presence accuracy: `1.000000`
- present type accuracy: `1.000000`
- normalized vertex MAE: `7.786439e-07`
- pred / target area: `189` / `189`

S265 confirms that the S262 failure came from vertex-to-hard-raster sensitivity: the failed run had target-vertex oracle IoU `1.000000`, but predicted vertices expanded the raster mask by `25` false-positive pixels. S267 reduces the largest grid-cell vertex error from roughly `0.33 / 0.49` x/y cells to `0.003 / 0.006` x/y cells.

## Decision

The polygon inverse route should continue. The next stage may resume the original 5-sample overfit gate, using the same supervised polygon inverse runner and keeping hard polygon rasterization as evaluation-only.

No polygon inverse candidate is promoted, the S185/S181 center-bin branch candidate is unchanged, and this remains separate from any main baseline replacement.
