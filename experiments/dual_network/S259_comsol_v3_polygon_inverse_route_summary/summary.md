# S259 COMSOL V3 Polygon Inverse Route Summary

S259 starts the supervised polygon inverse route after S254-S258 proved that the polygon-compatible repaired V3 pack passes signal, mask, target, and hard polygon oracle gates.

## Route

- Input: multi-height repaired Bz signals.
- Output: fixed-slot polygon predictions.
- Target: normalized four-corner polygon vertices.
- Evaluation: hard polygon rasterization IoU.

## Boundaries

- Do not train the old S185/S181 center-bin candidate.
- Do not replace the current branch candidate.
- Do not claim a main baseline replacement.
- Do not generate new COMSOL data.
- Do not introduce differentiable polygon raster loss in this first gate.

## Decision

Use a new polygon inverse model and runner instead of extending `train_comsol_parametric_inverse.py`. The existing parametric runner is tied to `center + axis + rotation` targets and metrics; polygon vertices need separate model, loss, export, and mask-evaluation semantics.
