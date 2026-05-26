# S284 COMSOL V3 Polygon Generalization Failure Summary

S284 starts a no-training diagnostic stage after S279-S283 repaired the polygon inverse train30 fit gate.

## Input State

`longer_train30` passes train fit:

- train mean/min polygon IoU: `0.935101` / `0.802920`
- train presence/type accuracy: `1.000000` / `1.000000`
- train vertex MAE: `5.560893e-05`

Held-out observation remains weak:

- val mean/min polygon IoU: `0.046352` / `0.000000`
- test mean/min polygon IoU: `0.136720` / `0.000000`
- val/test vertex MAE: `6.150539e-03` / `3.714611e-03`

## Boundary

This stage does not run training, multi-seed validation, larger models, extra steps, or new COMSOL data. The goal is to explain the held-out failure mode before designing any new training experiment.

This result does not replace the S185/S181 center-bin branch candidate and is not a main baseline replacement.
