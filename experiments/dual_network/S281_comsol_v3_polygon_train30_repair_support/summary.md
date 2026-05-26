# S281 COMSOL V3 Polygon Train30 Repair Support

S281 confirms that no runner or model code change is needed for this repair stage.

## Existing Support

`train_comsol_polygon_inverse.py` already supports the knobs needed for S282:

- longer training through `--steps`
- larger capacity through `--hidden-dim` and `--latent-dim`
- default-off `--lambda-center-aux`, `--lambda-box-aux`, `--lambda-area-aux`, and `--lambda-edge-aux`
- `--vertex-loss-space norm|grid`
- prediction export for hard polygon diagnostics

## Decision

Do not modify Python in this stage. The repair gate can run with existing runner behavior, preserving the S259-S278 polygon inverse semantics and smoke-test assumptions.

No Claude Code review is needed because no model, runner, target, or rasterizer code is changed.
