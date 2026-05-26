# S236 COMSOL V3 repaired ingest gate decision

S233-S235 pass the repaired V3 hard-case ingest gate.

The repaired pack is usable for the next branch-local evaluation stage because:

- `defect_params.csv` distribution was re-counted from the ingested files and matches sample counts for train/val/test.
- converted shapes are `[30,3,200]`, `[10,3,200]`, and `[10,3,200]`.
- repaired Bz signals are finite and comfortably above the `1e-8` runner floor.
- `masks == (mu_maps < 500)` mismatch is `0` on all splits.
- fixed-order parametric targets are built with schema `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`.
- oracle rasterization reaches train/val/test IoU `1.000000` / `1.000000` / `1.000000`.

Decision: proceed to repaired V3 candidate evaluation as a new stage. This stage does not train, does not change model structure, does not change runner defaults, and does not promote any result to a main baseline replacement.

Boundary: the repaired V3 fallback pack is still single unrotated Block `rectangular_notch` COMSOL data, not true rotated or multi-component COMSOL geometry coverage.
