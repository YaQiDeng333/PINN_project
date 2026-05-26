# S161 center diagnostics split summary: train

## Grid and units

- geometry_units: `m`
- dx_grid: `4.020100e-04`
- dy_grid: `2.020202e-04`
- samples: `100`
- present_components: `300`

## Aggregate center errors

- center_x_grid_mae: `8.013843e-01`
- center_y_grid_mae: `3.639531e-01`
- center_l2_grid_mae: `9.402238e-01`
- center_x_axis_relative_mae: `6.613034e-02`
- center_y_axis_relative_mae: `1.269992e-02`
- center_axis_relative_l2_mae: `6.899444e-02`

## Correlation with mask IoU

- center_l2_grid_mae: Pearson=-2.343548e-01, Spearman=-1.747015e-01; center_axis_relative_l2_mae: Pearson=-2.269104e-01, Spearman=-1.588959e-01

## Interpretation

- worst component_slot by center_l2_grid_mae: `1` = `1.003111e+00`.
- recommended center loss mode: `grid_mse`.
- Negative correlation means larger center error tends to lower mask IoU.
