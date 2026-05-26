# S310 bounded local output support

Implemented default-off bounded local vertex output in `train_comsol_center_anchored_polygon_inverse.py`.

New CLI:

- `--local-shape-output-mode raw|bounded_tanh`, default `raw`;
- `--local-shape-bound-mode fixed_grid|train_stats`, default `fixed_grid`;
- `--local-shape-fixed-bound-x-grid 24.0`;
- `--local-shape-fixed-bound-y-grid 8.0`;
- `--local-shape-train-stats-margin 1.25`.

Mechanics:

- `raw` mode preserves the previous behavior exactly: the model head output is used as `local_vertices_grid`.
- `bounded_tanh` mode maps the raw local head through `tanh(raw) * [bound_x, bound_y]`.
- `fixed_grid` uses `[24, 8]` grid-cell bounds.
- `train_stats` computes bounds from present train targets only, using `max_abs_local_x/y * 1.25`; val/test targets are not used to set bounds.
- The effective local vertices are used consistently for local vertex loss, vertex decode, hard polygon raster metrics, and prediction export.

The center-bin path is unchanged. Existing y-bin extra-loss options from S303-S307 remain available but are not enabled in S311.
