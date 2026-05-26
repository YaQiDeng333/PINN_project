# S177 COMSOL center-bin offset support

## Implementation summary

S177 adds optional center-bin + offset support to the existing COMSOL parametric inverse runner.

New model / runner mode:

- `--center-representation continuous|bin_offset`
- `--center-bin-size-cells`
- `--lambda-center-bin`
- `--lambda-center-offset`

Default behavior remains `continuous`, preserving the old output dictionary and existing baseline behavior.

## Bin-offset mode

In `bin_offset` mode the model predicts:

- `center_x_bin_logits`
- `center_y_bin_logits`
- `center_offset`

The decoded center replaces `center_x` / `center_y` in the effective continuous tensor used for:

- center-grid loss
- raster loss
- evaluation metrics
- prediction export
- hard rasterization

Axis, depth, and rotation still come from the continuous head. The base continuous loss skips center dimensions in `bin_offset` mode, because center supervision comes from bin CE, offset SmoothL1, and optional decoded center-grid loss.

## Review

Claude Code review found no must-fix correctness bugs. A non-blocking observability suggestion was addressed by adding bin accuracy / offset MAE metrics and a smoke check that decoded centers are finite and remain within a broad grid range.

## Self-review

Default compatibility is preserved, `permutation_min` is rejected for `bin_offset`, and the implementation keeps decoded center use consistent across loss / eval / export / raster paths.
