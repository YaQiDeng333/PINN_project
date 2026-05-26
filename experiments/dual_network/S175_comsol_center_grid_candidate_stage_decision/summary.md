# S175 COMSOL center-grid candidate stage decision

## Final decision

S171-S175 completes documentation consolidation for the current COMSOL parametric candidate:

`raw MLP / shared head / fixed-order + lambda_center_grid=0.1`

This is the current candidate only for the `feature/dual-network-variational` branch. It is not a main baseline replacement.

## What changed in this stage

- Added documentation summaries for S171-S175.
- Added a reproducible candidate command to `DUAL_NETWORK_REPRODUCE.md`.
- Updated README, experiment log, stage summary, terms, and artifact index.

## What did not change

- No Python files were modified.
- No CLI default was changed.
- No training was run.
- No new seed was added.
- No py_compile was needed.
- No checkpoint, model weight, image, `.npy`, or raw export artifact was created.

## Next step

Use the consolidated center-grid candidate as the reference for the next route. The next recommended route is `center-bin classification + offset`.

## Self-review

S175 closes a documentation-only consolidation stage. The result is decision-complete for future comparison without expanding experiment cost.
