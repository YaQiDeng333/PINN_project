# S260 COMSOL V3 Polygon Inverse Model Skeleton

S260 adds an independent polygon inverse model.

## Model

- file: `comsol_polygon_inverse_models.py`
- input: flattened per-sample-zscored Bz signal `[B,600]`
- output:
  - `presence_logits [B,3]`
  - `presence_prob [B,3]`
  - `type_logits [B,3,T]`
  - `vertices_norm [B,3,4,2]`
- first-stage constraints: `max_components=3`, `max_vertices=4`, fixed slot, `clockwise_top_left` vertex ordering.

## Smoke

`smoke_test_comsol_polygon_inverse_models.py` checks output shapes, finite loss, backward pass, and fixed-shape guards. The smoke test passed.

This stage does not modify the existing parametric inverse model or runner.
