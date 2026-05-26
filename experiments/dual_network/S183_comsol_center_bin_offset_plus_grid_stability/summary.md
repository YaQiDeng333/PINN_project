# S183 COMSOL center-bin offset plus grid stability repeat

## Runs

S183 reused the S179 seed1 run and added seed2 / seed3 with the same 3000-step configuration:

- raw MLP signal encoder
- shared parametric head
- fixed component order
- `center_representation=bin_offset`
- `center_bin_size_cells=8`
- `lambda_center_bin=1.0`
- `lambda_center_offset=1.0`
- `lambda_center_grid=0.1`
- `lambda_center_axis_relative=0.0`
- no raster loss
- no forward consistency
- no validation-aware endpoint selection

## Per-run metrics

| run | seed | train IoU | val IoU | test IoU | val center_grid_mae | test center_grid_mae |
|---|---:|---:|---:|---:|---:|---:|
| S179 reused seed1 | 1 | 0.716101 | 0.542935 | 0.581320 | 3.362513 | 2.721649 |
| S183 seed2 | 2 | 0.725698 | 0.484303 | 0.575504 | 6.282760 | 2.929023 |
| S183 seed3 | 3 | 0.726279 | 0.492127 | 0.578738 | 6.026593 | 2.804331 |

## Gate status

Seed2 did not trigger the early stop condition: val/test IoU stayed above the S170 center-grid range lower bound, and val/test `center_grid_mae` stayed below the S170 worst values. Seed3 was therefore executed.

## Self-review

The repeat provides three completed runs for the S184 aggregate judgment. Seed2/seed3 have lower val IoU than S179 seed1, so any promotion should keep an explicit observation caveat.
