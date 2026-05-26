# S266 COMSOL V3 Polygon Loss Repair Support

S266 adds default-off polygon inverse loss support for the one-sample repair gate. The model structure is unchanged.

## Runner Changes

- `--vertex-loss-space norm|grid`, default `norm`.
- `--lambda-area-aux`, default `0.0`.
- `--lambda-edge-aux`, default `0.0`.

`norm` preserves the S259-S263 vertex SmoothL1 behavior. `grid` scales x/y vertex coordinates by the corresponding raster grid spacing before SmoothL1, so the vertex loss penalizes errors in grid-cell units rather than raw normalized units.

The area auxiliary uses torch shoelace polygon area in grid-cell units. The edge auxiliary uses the four edge lengths in grid-cell units. Both only apply to present components and remain disabled unless their lambda is non-zero.

The area/edge auxiliaries intentionally require all four vertices to be valid for every present component. The current polygon route is fixed four-corner geometry; rejecting 3-vertex present polygons avoids silently training area or closing-edge losses against padded vertices.

## Boundary

S266 does not add a differentiable polygon rasterizer and does not change default behavior. Hard polygon rasterization remains an evaluation/export metric.
