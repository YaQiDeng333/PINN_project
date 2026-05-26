# S255 COMSOL V3 Polygon Converted Validation

The S254 raw pack was converted with `convert_comsol_multiheight_csv_to_npz.py` into V2-compatible multi-height NPZ files.

## Converted Shapes

| split | converted NPZ | signals shape |
| --- | --- | --- |
| train | `S254_comsol_v3_polygon_hard_case_ingest/converted/train_comsol_v3_polygon_hard_case.npz` | `[30,3,200]` |
| val | `S254_comsol_v3_polygon_hard_case_ingest/converted/val_comsol_v3_polygon_hard_case.npz` | `[10,3,200]` |
| test | `S254_comsol_v3_polygon_hard_case_ingest/converted/test_comsol_v3_polygon_hard_case.npz` | `[10,3,200]` |

## Validation

- finite signal values: pass
- complete per-sample/channel `x_index=0..199`: pass
- `masks == (mu_maps < 500)` mismatch: `0`
- normalized x range: `[-0.04,0.04]`
- normalized y range: `[-0.01,0.01]`
- embedded polygon fields present: `polygon_vertices_raw`, `polygon_vertices_norm`, `polygon_vertex_mask`, `polygon_presence`
- hard-case distribution matches `defect_params.csv`.

This stage performs ingest validation only; no training or candidate replacement is performed.
