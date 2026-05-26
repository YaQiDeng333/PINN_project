# S263 COMSOL V3 Polygon Inverse Decision

S259-S263 implement the first supervised polygon inverse gate. The independent model and runner smoke tests pass, but the one-sample overfit gate does not pass the required hard polygon mask IoU threshold.

## Decision

- Do not run 5-sample or train30 polygon inverse training yet.
- Do not promote any polygon inverse candidate.
- Do not replace the S185/S181 center-bin candidate.
- Do not interpret this as polygon target failure; S257 oracle IoU is still `1.000000`.

## Most Likely Issue

The one-sample model predicts the right component presence/type and very close vertices, but hard raster IoU remains `0.883178`. The next stage should diagnose vertex-to-raster sensitivity before increasing experiment scale.

## Next Step

Recommended next route: add a one-sample polygon vertex/raster diagnostic to compare target vertices, predicted vertices, rasterized area, edge displacement in grid-cell units, and whether center/box auxiliary loss or coordinate scaling is needed before retrying tiny-overfit.
