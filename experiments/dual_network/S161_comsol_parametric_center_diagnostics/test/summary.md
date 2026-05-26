# S161 center diagnostics split summary: test

## Grid and units

- geometry_units: `m`
- dx_grid: `4.020100e-04`
- dy_grid: `2.020202e-04`
- samples: `20`
- present_components: `60`

## Aggregate center errors

- center_x_grid_mae: `3.832278e+00`
- center_y_grid_mae: `5.097501e+00`
- center_l2_grid_mae: `6.998191e+00`
- center_x_axis_relative_mae: `3.157011e-01`
- center_y_axis_relative_mae: `1.761601e-01`
- center_axis_relative_l2_mae: `3.942146e-01`

## Correlation with mask IoU

- center_l2_grid_mae: Pearson=-9.106166e-01, Spearman=-9.082707e-01; center_axis_relative_l2_mae: Pearson=-9.669712e-01, Spearman=-9.593985e-01

## Interpretation

- worst component_slot by center_l2_grid_mae: `0` = `7.609438e+00`.
- recommended center loss mode: `axis_relative_smoothl1`.
- Negative correlation means larger center error tends to lower mask IoU.
