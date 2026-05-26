# S248 COMSOL V3 polygon export schema request

S248 defines the COMSOL-side export contract for true polygon geometry. The canonical request is `COMSOL_V3_POLYGON_GEOMETRY_DATA_REQUEST.md`.

## Required file

`polygon_params.csv` must contain one row per sample, component, and vertex:

```text
sample_index,split,component_slot,component_id,component_type,vertex_index,x_raw,y_raw,x_norm,y_norm,ordering,geometry_feature_tag,selection_name,hard_case_type,component_count,union_selection_name,true_rotated_geometry,true_multi_component_geometry
```

## Required semantics

- Vertices must come from the actual COMSOL geometry used in the solve.
- Raw and normalized coordinates are both required.
- Vertex ordering is fixed as clockwise with start at the normalized top-left-like corner.
- `true_rotated_geometry` and `true_multi_component_geometry` are audit fields, not labels that can be written without matching solved geometry.
- Multi-component samples must use a COMSOL Union or equivalent solved multi-feature geometry.

## Stop condition

If COMSOL cannot export actual corner points or if rotated/multi-component geometry exists only in metadata, the polygon route stops before model training.
