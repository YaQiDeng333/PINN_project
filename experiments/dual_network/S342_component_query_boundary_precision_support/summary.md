# S342 Component-Query Boundary Precision Support

S342 keeps the existing component-query model unchanged and only exposes existing default-off loss knobs needed for the planned 1-sample gate.

Support changes:

- Reuse `--lambda-decoded-center-aux`, already defaulting to `0.0`.
- Expose `--lambda-area-aux` in `train_comsol_component_query_polygon_inverse.py`, default `0.0`.
- Preserve old default behavior and old center-anchored runner behavior.

No raster loss, y-loss, bound sweep, local-conditioning sweep, edge-only aux, larger model, or extra training steps are introduced.
