# S256 COMSOL V3 Polygon Target Validation

S256 exports rasterizer-ready polygon targets from the converted NPZ embedded polygon arrays plus the wide `polygon_params.csv` audit table.

## Target Schema

- `polygon_vertices_raw`: `[N,3,4,2]`
- `polygon_vertices_norm`: `[N,3,4,2]`
- `polygon_vertex_mask`: `[N,3,4]`
- `presence_targets`: `[N,3]`
- `type_targets`: `[N,3]`
- `component_counts`: `[N]`
- `sample_indices`: `[N]`
- `type_vocab`: `rectangular_notch, rotated_rect`
- `vertex_ordering`: `clockwise_top_left`

## Split Coverage

| split | samples | polygon component rows | true rotated rows | true multi-component rows |
| --- | ---: | ---: | ---: | ---: |
| train | `30` | `38` | `22` | `15` |
| val | `10` | `13` | `7` | `6` |
| test | `10` | `13` | `8` | `6` |

The utility verifies that `polygon_params.csv` covers every split-local `sample_index=0..N-1` and that each sample's polygon row count matches `polygon_presence`.
