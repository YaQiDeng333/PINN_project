# S251 COMSOL V3 polygon true-geometry 3-sample smoke

S251 runs a 3-sample real COMSOL smoke for the polygon route. It does not generate a full pack and does not train.

## Smoke samples

| sample | hard_case_type | geometry |
|---:|---|---|
| 0 | high_angle_rotated_single | one true rotated Block, rotation `25 deg` |
| 1 | two_component_union | true two-component Union with one rotated component and one axis-aligned component |
| 2 | narrow_near_boundary_corner_safe | narrow rotated component near boundary, still corner-safe |

All samples use a fresh COMSOL model and the repaired near-defect reduced-field Bz signal route (`mfnc.redBz`). Polygon vertices are exported in raw COMSOL coordinates and normalized V2-compatible coordinates.

## Solver and signal checks

| sample | std min | std max | peak-to-peak min | peak-to-peak max |
|---:|---:|---:|---:|---:|
| 0 | `1.516766520233e-06` | `1.556078489285e-06` | `1.056513578251e-05` | `1.083980211387e-05` |
| 1 | `4.022578918109e-07` | `4.033053448916e-07` | `3.229000379097e-06` | `3.304150151680e-06` |
| 2 | `2.147334448539e-06` | `2.164347108382e-06` | `1.313836631536e-05` | `1.314402114426e-05` |

All three solves succeeded. The local raw smoke output is `comsol_v3_polygon_geometry_3sample_smoke/`; it is not staged because it is generated COMSOL smoke data.

## Boundary

The first attempted high-angle geometry was too aggressive for the solver. The final smoke keeps true rotation and true multi-component coverage while using a more conservative, stable geometry. This isolates schema/rasterizer validation from solver divergence.
