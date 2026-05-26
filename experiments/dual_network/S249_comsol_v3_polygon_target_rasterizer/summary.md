# S249 COMSOL V3 polygon target builder and hard rasterizer

S249 adds a small polygon target/rasterizer skeleton without touching the parametric inverse training runner.

## Added scripts

- `comsol_polygon_targets.py`
- `comsol_polygon_rasterizer.py`
- `smoke_test_comsol_polygon_rasterizer.py`

## Target builder

`comsol_polygon_targets.py` reads a COMSOL-style `polygon_params.csv` and an NPZ containing `masks`, `x`, and `y`. It writes `polygon_targets.npz` with:

- `polygon_vertices_raw [N, max_components, max_vertices, 2]`
- `polygon_vertices_norm [N, max_components, max_vertices, 2]`
- `polygon_vertex_mask [N, max_components, max_vertices]`
- `presence_targets [N, max_components]`
- `type_targets [N, max_components]`
- `component_counts`, `sample_indices`, `type_vocab`, and `metadata_json`

The builder enforces clockwise ordering with start at the normalized-space top-left-like vertex.

## Hard rasterizer

`comsol_polygon_rasterizer.py` rasterizes present component polygons with point-in-polygon semantics and combines components by boolean OR union. It supports raw or normalized vertex space. The first-stage oracle gate uses normalized vertices.

No differentiable polygon rasterizer is added in this stage. Differentiable raster / mask fine-tuning is deferred until the hard oracle path is reliable.
