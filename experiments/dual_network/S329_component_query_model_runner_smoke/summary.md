# S329 Component-Query Model / Runner Smoke Summary

新增 component-query model 和 independent runner 后，先完成 smoke gate，再进入任何 overfit 实验。

## Checks

- `python smoke_test_comsol_component_query_polygon_inverse_models.py`
  - 覆盖 `B=5`, `signals=600`, `Q=3`, `V=4` forward shape。
  - 覆盖 finite supervised loss 和 backward。
  - 确认 query embedding 与 output heads 有梯度。
- `python smoke_test_train_comsol_component_query_polygon_inverse.py`
  - 使用 tempfile 小包运行极短训练。
  - 检查 `metrics.csv`, `training_history.csv`, `run_summary.md`, prediction export。
  - 检查没有保存 checkpoint / weights / `.npy`。

## Result

S329 smoke 通过。component-query runner 可读 center-anchored polygon targets，并复用现有 hard decode / polygon rasterizer evaluation 语义。旧 runner 默认行为未修改。

## Next Gate

进入 S330 1-sample gate。若 1-sample IoU `<0.99`，停止，不进入 5-sample 或 train30。
