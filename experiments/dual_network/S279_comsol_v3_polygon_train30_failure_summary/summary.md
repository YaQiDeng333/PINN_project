# S279 COMSOL V3 Polygon Train30 Failure Summary

S279 starts a train-fit repair stage for the COMSOL V3 polygon inverse route. The goal is only to repair train30 fitting before any multi-seed, larger-data, or val/test generalization discussion.

## Input State

- S257 polygon oracle remains `1.000000`.
- S267 one-sample overfit passed.
- S271 five-sample overfit passed.
- S275 train30 failed the train-fit gate:
  - train mean/min polygon IoU: `0.731445` / `0.518519`
  - train presence/type accuracy: `1.000000` / `1.000000`
  - train vertex MAE: `1.793932e-04`
  - val/test mean IoU observation only: `0.033122` / `0.089484`

## Interpretation

The failure is not a polygon target, vertex ordering, hard rasterizer, presence, or type issue. It is a train30-scale vertex precision issue: the model can fit small subsets, but the same configuration does not fit all 30 train hard cases tightly enough for hard polygon mask IoU.

## Boundary

- Do not run multi-seed validation.
- Do not expand data.
- Do not discuss val/test generalization until train30 fit passes.
- Do not replace the S185/S181 center-bin branch candidate.
- Do not write this as a main baseline replacement.
