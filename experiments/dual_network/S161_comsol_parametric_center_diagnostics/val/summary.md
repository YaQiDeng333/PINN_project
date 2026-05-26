# S161 center diagnostics split summary: val

## Grid and units

- geometry_units: `m`
- dx_grid: `4.020100e-04`
- dy_grid: `2.020202e-04`
- samples: `20`
- present_components: `60`

## Aggregate center errors

- center_x_grid_mae: `4.375079e+00`
- center_y_grid_mae: `5.998688e+00`
- center_l2_grid_mae: `8.017750e+00`
- center_x_axis_relative_mae: `3.641928e-01`
- center_y_axis_relative_mae: `2.031887e-01`
- center_axis_relative_l2_mae: `4.450618e-01`

## Correlation with mask IoU

- center_l2_grid_mae: Pearson=-9.265284e-01, Spearman=-9.022556e-01; center_axis_relative_l2_mae: Pearson=-9.728010e-01, Spearman=-9.744361e-01

## Interpretation

- worst component_slot by center_l2_grid_mae: `0` = `8.600240e+00`.
- recommended center loss mode: `axis_relative_smoothl1`.
- Negative correlation means larger center error tends to lower mask IoU.
