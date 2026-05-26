# S315 local-shape conditioning smoke

Smoke checks required before S316:

- model forward/backward covers `none` and `center_bin_slot_type`;
- runner smoke covers default behavior, y-bin extra loss, bounded local output, train-stats bounds, and `center_bin_slot_type` conditioning;
- py_compile covers the model, runner, and both smoke tests.

Initial check commands:

- `python smoke_test_comsol_center_anchored_polygon_inverse_models.py`
- `python smoke_test_train_comsol_center_anchored_polygon_inverse.py`
- `python -m py_compile comsol_center_anchored_polygon_inverse_models.py train_comsol_center_anchored_polygon_inverse.py smoke_test_comsol_center_anchored_polygon_inverse_models.py smoke_test_train_comsol_center_anchored_polygon_inverse.py`

All checks must pass before running S316 quick-gate training.

Executed with `C:\Users\19166\anaconda3\envs\comsol_mcp\python.exe`; model smoke, runner smoke, and py_compile passed before S316 quick-gate training.

Codex subagent review found no critical or important issues. Its only minor request was to assert the central detach property directly, so the model smoke now also verifies that local-only vertex loss does not backpropagate into `center_x_bin_head`, `center_y_bin_head`, `center_offset_head`, or `type_head` through conditioning. The updated smoke and py_compile pass.
