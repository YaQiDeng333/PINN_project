# S193 COMSOL signal-to-center auxiliary decision

## Summary

S189-S193 implemented and quick-gated an optional signal-to-center auxiliary head. The auxiliary head is technically usable, but the S191 quick gate did not show held-out improvement over the same-round current-candidate reference.

## Decision

The auxiliary head is not promoted and does not enter full confirm in this stage.

Current COMSOL parametric route candidate remains:

- raw MLP;
- shared head;
- fixed-order components;
- `center_representation=bin_offset`;
- `center_bin_size_cells=8`;
- `lambda_center_bin=1.0`;
- `lambda_center_offset=1.0`;
- `lambda_center_grid=0.1`;
- `lambda_center_axis_relative=0.0`;
- no raster loss;
- no forward consistency;
- no validation-aware endpoint selection.

This is still a `feature/dual-network-variational` branch candidate, not a main baseline replacement.

## Interpretation

The auxiliary head improved training fit but did not improve the final decoded center/rasterized mask metrics on held-out splits. In particular, the x-weighted auxiliary variant did not convert better x-bin emphasis into a val/test IoU gain.

This suggests the remaining instability is not solved by simply adding another center-bin/offset prediction head to the same latent with comparable supervision. The next route should avoid more weight tuning on this auxiliary form.

## Next Step

Recommended next direction:

- diagnose whether the signal latent and main center head disagree on the same samples, then consider a more structural center representation refinement only if that diagnostic points to a clear failure mode.

Not recommended now:

- promoting the auxiliary head;
- increasing auxiliary lambda values;
- returning to raster loss, forward consistency, type/rotation loss, or dense conditional mask runner sweeps.

## Self-Review

- S193 does not upgrade the auxiliary head to the current candidate.
- The decision is based on same-round reference comparison.
- S192 was correctly skipped under the stated gate.
