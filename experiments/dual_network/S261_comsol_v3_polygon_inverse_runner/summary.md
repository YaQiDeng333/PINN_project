# S261 COMSOL V3 Polygon Inverse Runner

S261 adds an independent polygon inverse runner for supervised vertex regression.

## Runner

- file: `train_comsol_polygon_inverse.py`
- reads converted NPZ signals/masks/x/y plus `polygon_targets.npz`
- uses per-sample z-score signal normalization
- trains full-batch Adam without saving checkpoints or weights

## Loss

- presence: BCE over all fixed slots
- type: CE only over present slots
- vertex: SmoothL1 over present slots and valid vertices
- optional center/box auxiliary losses are available but default to `0.0`
- hard polygon rasterizer is used only for evaluation, not training loss

## Outputs

- `metrics.csv`
- `eval_metrics.csv`
- `test_metrics.csv`
- `training_history.csv`
- `run_summary.md`
- optional `*_polygon_predictions.csv` and `*_polygon_mask_metrics.csv` with `--export-predictions`

`smoke_test_train_comsol_polygon_inverse.py` passed.
