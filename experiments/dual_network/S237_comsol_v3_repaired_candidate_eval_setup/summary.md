# S237 COMSOL V3 repaired candidate evaluation setup

本阶段评估 repaired COMSOL V3 hard-case fallback pilot，不修改训练 runner，不修改模型结构，不 push，也不把结果写成 main baseline replacement。

## Data

- Repaired V3 train/val/test converted shapes: `[30,3,200]`, `[10,3,200]`, `[10,3,200]`
- Hard-case distribution from S233 `defect_params.csv`: train `10/5/7/5/3`, val `3/2/2/2/1`, test `3/2/2/2/1`
- Signal std and peak-to-peak are non-degenerate and above the `1e-8` runner floor.
- `masks == (mu_maps < 500)` mismatch is `0`.
- Oracle rasterization IoU is `1.000000` / `1.000000` / `1.000000`.

## Candidate

The evaluated branch candidate is S185/S181 `center_bin_offset_plus_grid`:

- `encoder_type=mlp`
- `head_mode=shared`
- `component_matching_mode=fixed`
- `center_representation=bin_offset`
- `center_bin_size_cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no val selection

Boundary: this repaired pack is still a single unrotated Block `rectangular_notch` fallback pilot, not true rotated or multi-component COMSOL geometry.
