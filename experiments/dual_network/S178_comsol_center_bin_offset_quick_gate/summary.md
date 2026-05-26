# S178 COMSOL center-bin offset quick gate

## Configuration

All runs use:

- S84 converted V2 NPZ + S113 raw targets
- `steps=1500`
- `seed=1`
- raw MLP encoder
- shared head
- fixed-order components
- no raster loss
- no forward consistency
- no validation selection
- prediction export enabled

## Results

| run | train IoU | val IoU | test IoU | val center_grid_mae | test center_grid_mae |
|---|---:|---:|---:|---:|---:|
| current_candidate_reference_1500_seed1 | 0.724849 | 0.494508 | 0.493461 | 5.369309 | 4.907973 |
| center_bin_offset | 0.720732 | 0.496768 | 0.517553 | 6.176620 | 5.836287 |
| center_bin_offset_plus_grid | 0.704860 | 0.546311 | 0.586546 | 3.250844 | 2.572883 |

## Gate decision

`center_bin_offset_plus_grid` passed the quick gate:

- val IoU improved by `+0.051803` over the same-round current candidate reference.
- test IoU improved by `+0.093085`.
- val/test center_grid_mae both decreased.
- presence stayed at `1.0`.

`center_bin_offset` without grid loss improved test IoU but worsened center_grid_mae and did not clearly improve val, so it is not selected for full confirm.

## Self-review

The gate uses the same-round current candidate reference, not the historical param-only baseline. S178 supports running S179 full confirm for `center_bin_offset_plus_grid`.
