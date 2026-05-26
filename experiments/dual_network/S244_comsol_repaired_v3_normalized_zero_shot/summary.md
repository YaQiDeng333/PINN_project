# S244 repaired COMSOL V3 normalized zero-shot evaluation

S244 reruns the V2-train to repaired V3 val/test path after S242 normalization. The previous `center_x target is outside the x grid range` error is gone; this is now a real zero-shot evaluation, not just a runability gate.

## Setup

- Train data: V2 train converted NPZ and S113 train parametric targets.
- Eval data: S242 normalized repaired V3 val/test converted NPZ and S243 normalized targets.
- Candidate: raw MLP / shared head / fixed-order, `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, `lambda_center_grid=0.1`, no raster loss, no forward consistency, no val selection.
- `seed=1`, `steps=3000`, `--export-predictions`.

## Metrics

| split | mask IoU | Dice | center_grid_mae | x_bin_acc | y_bin_acc | center_offset_mae | presence_acc | type_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.007616 | 0.014663 | 50.908295 | 0.100000 | 0.100000 | 0.282466 | 0.333333 | 0.600000 |
| test | 0.005248 | 0.009972 | 58.765858 | 0.000000 | 0.100000 | 0.255382 | 0.333333 | 0.700000 |

## Interpretation

The coordinate convention mismatch is fixed, but V2-trained zero-shot transfer to the repaired V3 fallback pilot remains very weak. The failure is dominated by center-bin and presence errors, not by near-constant signals or target/raster mismatch.
