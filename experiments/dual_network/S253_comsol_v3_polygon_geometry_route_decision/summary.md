# S253 COMSOL V3 polygon geometry route decision

S247-S252 pass the first polygon-route gate.

## Decision

Proceed with the polygon / corner-point geometry route for V3 true rotated and multi-component COMSOL data. The recommended V3 true-geometry representation is fixed four-corner polygons with `max_components=3` and `max_vertices=4`, using normalized vertices as the training/oracle representation and raw vertices as COMSOL audit truth.

## What this does not change

- The current S185/S181 `center_bin_offset_plus_grid` candidate remains the V2-style branch candidate.
- No training was run.
- No larger V3 pack was generated.
- No main baseline replacement is claimed.
- No dense runner, type/rotation loss sweep, raster loss sweep, or forward-consistency sweep is restarted.

## Next stage

The next stage should design a polygon inverse runner only after this oracle path remains stable on additional true COMSOL polygon smoke or a user-confirmed polygon pack generation stage. The first model route should use direct vertex regression with presence/type supervision; differentiable polygon raster or mask fine-tuning should remain a later step.

## Stop conditions for the next stage

- COMSOL cannot export true geometry vertices.
- Rotated or multi-component labels are metadata-only.
- Polygon oracle IoU falls below `0.95`.
- Vertex ordering or component slot alignment becomes ambiguous.
- COMSOL solver divergence requires expanding compute beyond smoke scale without user confirmation.
