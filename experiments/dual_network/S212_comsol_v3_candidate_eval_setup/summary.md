# S212 COMSOL V3 hard-case candidate evaluation setup

本阶段评估当前 COMSOL parametric branch candidate 在真实 COMSOL V3 hard-case fallback pilot 上的表现。阶段边界是评估和诊断，不替换 main baseline，不声称覆盖 rotated / multi-component 真实 COMSOL 几何，也不运行 dense conditional mask runner。

## Data

- V3 source: `experiments/dual_network/S208_comsol_v3_hard_case_ingest/`
- split sizes: train `30`, val `10`, test `10`
- converted shapes: train `[30,3,200]`, val `[10,3,200]`, test `[10,3,200]`
- hard-case types: `x_bin_wrong_like`, `both_bins_wrong_like`, `bins_correct_center_or_offset_bad`, `geometry_or_type_interaction`, `rare_y_bin_wrong`
- oracle rasterization IoU: train / val / test = `1.000000` / `1.000000` / `1.000000`

## Candidate Configuration

- raw MLP
- shared head
- fixed-order components
- `center_representation=bin_offset`
- `center_bin_size_cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no validation-aware endpoint selection

## Boundary

The V3 fallback pack is real COMSOL data, not synthetic and not copied from V2. It currently contains single rectangular Block hard cases only. It should be used as a branch-local hard-case pilot, not as a broad V3 benchmark or a main baseline replacement.
