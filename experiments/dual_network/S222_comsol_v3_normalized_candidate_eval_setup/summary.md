# S222 normalized V3 candidate evaluation setup

S222 starts the normalized V3 hard-case candidate evaluation stage. This stage uses only the S218/S219 normalized V3 data and does not use the earlier raw-coordinate S208/S209 targets for model evaluation.

## Data

- V3 normalized converted NPZ: `experiments/dual_network/S218_comsol_v3_geometry_normalized/converted/`
- V3 normalized parametric targets: `experiments/dual_network/S219_comsol_v3_normalized_parametric_targets/`
- V3 oracle rasterization after normalization: train / val / test IoU = `1.000000` / `1.000000` / `1.000000`
- normalized V3 sample counts: train `30`, val `10`, test `10`

## Candidate

The evaluated branch candidate is the S185 `center_bin_offset_plus_grid` configuration:

- raw MLP
- shared head
- fixed-order component slots
- `center_representation=bin_offset`
- `center_bin_size_cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no validation selection

## Boundary

This is still a `feature/dual-network-variational` branch evaluation. It is not a main baseline replacement, and the V3 hard-case pack remains a real COMSOL single rectangular Block fallback pilot. It does not yet cover true rotated or multi-component COMSOL geometry.
