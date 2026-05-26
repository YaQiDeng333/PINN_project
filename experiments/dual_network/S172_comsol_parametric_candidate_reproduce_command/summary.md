# S172 COMSOL parametric candidate reproduce command

## Stage purpose

S172 documents a reproducible command for the current COMSOL parametric route candidate. No run was executed in this stage.

## Reproduce command boundary

The command is intentionally explicit rather than changing CLI defaults:

- `--lambda-center-grid 0.1`
- `--lambda-center-axis-relative 0.0`
- `--seed <N>`
- `--lambda-raster-bce 0.0`
- `--lambda-raster-dice 0.0`
- `--val-selection-metric none`
- `--val-selection-interval 0`

It uses `train_comsol_parametric_inverse.py`; it does not use the forward-consistency runner and does not include forward-consistency loss.

## Decision

`DUAL_NETWORK_REPRODUCE.md` is the reproduction entrypoint for this candidate. The runner default remains unchanged so older baseline commands continue to reproduce historical behavior unless the center-grid parameter is explicitly passed.

## Self-review

S172 is documentation-only and creates no new training output. The command is explicit enough to avoid silently changing historical baselines.
