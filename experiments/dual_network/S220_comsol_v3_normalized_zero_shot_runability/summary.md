# S220 COMSOL V3 normalized zero-shot runability gate

S220 verifies only that V2 train to normalized V3 val/test can run through the current center-bin candidate path without the previous center-bin range error. This is not a performance evaluation.

## Run

- train: V2 S84 train NPZ + S113 train parametric targets
- val/test: S218 normalized V3 NPZ + S219 normalized V3 parametric targets
- config: raw MLP / shared head / fixed-order, `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, `lambda_center_grid=0.1`
- steps: `5`
- seed: `1`
- output: `experiments/dual_network/S220_comsol_v3_normalized_zero_shot_runability/v2_train_to_v3_normalized_val_test/`

## Result

The command completed and exported metrics / predictions. The previous `ValueError: center_x target is outside the x grid range` did not occur.

## Boundary

This stage only proves that the coordinate convention has been normalized enough for the current runner to execute. The observed IoU or center metrics are not interpreted here. Performance evaluation is deferred to the next stage.
