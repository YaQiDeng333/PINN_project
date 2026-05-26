# S254 COMSOL V3 Polygon Hard-Case Raw Ingest

This stage copies the real COMSOL polygon-compatible repaired Bz hard-case pack into the branch experiment tree without training and without staging the source export root.

## Source

- source root: `comsol_geometry_v3_polygon_repaired_bz_hard_case_exports/`
- ingested copy: `experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/raw/`
- samples: train `30`, val `10`, test `10`
- signal rows: train `18000`, val `6000`, test `6000`
- route: repaired near-defect anomaly / reduced-field Bz
- geometry: true rotated and true multi-component COMSOL geometry are present in the solved pack

## Signal Stats

| split | signal std range | peak-to-peak range |
| --- | --- | --- |
| train | `7.106783e-07` - `2.953855e-06` | `2.960235e-06` - `2.238378e-05` |
| val | `9.916132e-07` - `3.365070e-06` | `8.472110e-06` - `1.906584e-05` |
| test | `8.530819e-07` - `2.740429e-06` | `4.030992e-06` - `2.005371e-05` |

All values are finite, every sample/channel has complete `x_index=0..199`, and `masks == (mu_maps < 500)` mismatch is `0`.

## Hard-Case Distribution

Order: `x_bin_wrong_like`, `both_bins_wrong_like`, `bins_correct_center_or_offset_bad`, `geometry_or_type_interaction`, `rare_y_bin_wrong`.

| split | distribution |
| --- | --- |
| train | `10 / 5 / 7 / 5 / 3` |
| val | `3 / 2 / 2 / 2 / 1` |
| test | `3 / 2 / 2 / 2 / 1` |

## Polygon Coverage

- total true rotated samples: `36`
- total true multi-component samples: `13`
- `polygon_params.csv` covers all present components for every split.
- Source export root remains outside the committed experiment tree.
