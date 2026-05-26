# S190 COMSOL signal-to-center auxiliary head support

## Implementation

S190 adds an optional auxiliary center head to the existing parametric inverse model and runner. No new runner was added.

The auxiliary head is disabled by default. When enabled with `--aux-center-head`, it requires `center_representation=bin_offset` and predicts:

- `aux_center_x_bin_logits [B,K,XB]`;
- `aux_center_y_bin_logits [B,K,YB]`;
- `aux_center_offset [B,K,2]`.

The auxiliary targets reuse the existing train-grid center-bin targets, bin-normalized offsets, fixed-order component slots, and present-component mask. The auxiliary loss is additive and does not replace the main center-bin/offset/grid losses.

New CLI parameters:

- `--aux-center-head`;
- `--lambda-aux-center-bin`;
- `--lambda-aux-center-offset`;
- `--aux-center-x-weight`;
- `--aux-center-y-weight`.

## Compatibility Boundary

- `--aux-center-head` defaults to off.
- The default `center_representation` remains unchanged.
- Existing lambda defaults remain unchanged.
- Existing main `center_*` metrics retain their meaning; auxiliary metrics use `aux_center_*` names.

## Checks

- `smoke_test_comsol_parametric_inverse_models.py` covers auxiliary output shapes, backward pass, default-off behavior, and invalid auxiliary use without bin-offset centers.
- `smoke_test_train_comsol_parametric_inverse.py` covers a tempfile auxiliary training run and verifies auxiliary history/metrics/summary fields.
- Claude Code review was attempted, but the CLI timed out before returning findings. Smoke tests and py_compile passed before continuing.

## Self-Review

- Auxiliary loss is gated by `--aux-center-head` and `center_representation=bin_offset`.
- The auxiliary head does not overwrite the main decoded center used for rasterized mask evaluation.
- x/y weights apply separately to bin CE and offset components.
