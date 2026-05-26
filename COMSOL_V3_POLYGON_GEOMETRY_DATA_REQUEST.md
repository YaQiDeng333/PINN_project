# COMSOL V3 polygon geometry data request

This request defines the geometry fields needed for the V3 polygon / corner-point route. The goal is to represent true COMSOL rotated and multi-component geometry directly, instead of forcing it through `center + axis + rotation`.

## Required polygon export

Create `polygon_params.csv` with one row per sample, component slot, and vertex:

```text
sample_index,split,component_slot,component_id,component_type,vertex_index,x_raw,y_raw,x_norm,y_norm,ordering,geometry_feature_tag,selection_name,hard_case_type,component_count,union_selection_name,true_rotated_geometry,true_multi_component_geometry
```

Required semantics:

- `x_raw`, `y_raw`: actual corner coordinates from the COMSOL geometry feature used in the solve.
- `x_norm`, `y_norm`: V2-compatible normalized coordinates using `x_norm=(x_raw-2250)*(0.08/4500)` and `y_norm=(y_raw-1500)*(0.02/3000)`.
- `ordering`: must be `clockwise_top_left`, with the first vertex chosen in normalized space by minimum `y`, then minimum `x`.
- `geometry_feature_tag`: COMSOL feature tag such as `blk1`.
- `selection_name`: COMSOL selection used by material/physics for that component or union.
- `union_selection_name`: non-empty when `component_count > 1`.
- `true_rotated_geometry`: true only if the solved COMSOL geometry feature is actually rotated.
- `true_multi_component_geometry`: true only if multiple solved COMSOL geometry features are present and unioned or equivalently solved.

## Guardrails

Metadata-only rotated or multi-component labels are not acceptable. If `true_rotated_geometry=true`, the exported vertices must not be axis-aligned unless the angle is exactly zero. If `component_count > 1`, the COMSOL solve must contain multiple geometry features or an equivalent union, not a single block plus multi-component CSV labels.

The first execution step should be a 3-sample polygon smoke, not a full pack:

1. high-angle single rotated Block;
2. two-component Union;
3. narrow / near-boundary but corner-safe component.

Each sample should use a fresh COMSOL model and the repaired near-defect reduced-field Bz route.
