# S250 COMSOL V3 polygon mock oracle smoke

S250 verifies the polygon target and hard rasterizer semantics on mock arrays before using COMSOL smoke data.

## Coverage

The smoke test covers:

- axis-aligned rectangle;
- rotated rectangle after non-uniform affine normalization;
- multi-component union;
- overlapping components;
- clockwise and counter-clockwise input ordering;
- degenerate polygon rejection.

## Result

`smoke_test_comsol_polygon_rasterizer.py` passed. The mock oracle validates that normalized polygon vertices can reconstruct target masks exactly for these controlled cases.

This is not COMSOL data and not a training result. It only validates target/rasterizer mechanics.
