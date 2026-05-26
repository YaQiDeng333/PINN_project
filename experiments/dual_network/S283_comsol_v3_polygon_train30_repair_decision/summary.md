# S283 COMSOL V3 Polygon Train30 Repair Decision

S279-S283 repairs the polygon inverse train30 fit gate.

## Decision

`longer_train30` passes the train30 fit gate:

- train mean/min polygon IoU: `0.935101` / `0.802920`
- train presence/type accuracy: `1.000000` / `1.000000`
- train vertex MAE: `5.560893e-05`
- all train hard-case groups have mean IoU `>= 0.75`

The previous S275 train30 failure was therefore a convergence / train-fit precision issue under hard-raster sensitivity, not a polygon schema failure.

## Boundary

- This does not promote a polygon inverse candidate.
- This does not replace the S185/S181 center-bin branch candidate.
- This is not a main baseline replacement.
- Val/test remain weak observation metrics and should not be interpreted as final polygon route generalization.

## Next Step

The next stage should plan polygon inverse held-out generalization diagnostics or controlled validation after the train-fit gate has been repaired. It should not start multi-seed candidate validation until the val/test failure mode is understood.
