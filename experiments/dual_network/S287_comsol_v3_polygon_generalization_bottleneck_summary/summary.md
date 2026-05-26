# S287 COMSOL V3 Polygon Generalization Bottleneck Summary

S287 combines the S285 distribution diagnostics and S286 prediction diagnostics.

## Distribution Diagnosis

The coarse split design is not obviously broken:

- hard-case distributions match the requested split design;
- true rotated rates are train/val/test `0.700` / `0.700` / `0.800`;
- true multi-component rates are `0.233` / `0.300` / `0.300`;
- signal std stays same-scale across splits: train/val/test mean `2.124018e-06` / `2.266323e-06` / `1.900533e-06`.

The main distribution caveat is sparse geometry coverage. Test is right-shifted in `center_x`: train/val/test means are `-0.001439` / `-0.002947` / `0.008342`. Test also has weaker left-side vertex-x coverage than train. This is not a hard schema error, but it is enough to matter with only `30/10/10` samples.

## Prediction Diagnosis

Train predictions are stable: no zero-IoU samples, no out-of-grid vertices, no signed-area flips, and vertex MAE stays near `5.6e-05`.

Val/test predictions are unstable: vertex MAE is two orders of magnitude larger than train, zero-IoU samples are common, and held-out polygons show signed-area flips and occasional out-of-grid vertices. Presence/type errors exist but do not explain most failures.

## Bottleneck

The most likely bottleneck is small-N memorization plus sparse geometry coverage causing held-out vertex/shape extrapolation failure. The current runner can fit train30, but the learned direct vertex mapping is not stable outside the exact train geometry set.

This is not evidence that the polygon target schema or repaired Bz signal route is invalid. It is also not enough to justify multi-seed validation, because the failure mechanism needs a targeted repair first.
