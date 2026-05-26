# S277 COMSOL V3 Polygon Train30 Quick Probe Decision

S274-S277 completes the first train30 polygon inverse quick probe.

## Decision

The train30 gate does not pass. Train mean/min polygon IoU are `0.731445` / `0.518519`, below the `0.90` / `0.80` gate. Presence/type accuracy is `1.000000`, so the model learns component existence and type but does not yet fit polygon vertices tightly enough across the full 30-sample train split.

## Boundary

- Do not run multi-seed validation.
- Do not promote a polygon inverse candidate.
- Do not replace the S185/S181 center-bin branch candidate.
- Do not interpret val/test as candidate failure beyond observation, because train fit is already below gate.

## Next Step

The next stage should be a targeted train-fit repair plan for polygon vertices. The likely routes are training-schedule/optimization repair, grid-space vertex loss, or area/edge auxiliary calibration; these should be planned before rerunning train30 or starting held-out candidate validation.
