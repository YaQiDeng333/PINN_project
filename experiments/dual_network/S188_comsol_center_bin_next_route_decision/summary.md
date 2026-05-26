# S188 COMSOL center-bin next route decision

## Decision

The next recommended route is `signal-to-center auxiliary head`.

## Rationale

- `center_bin_offset_plus_grid` is effective and is now the branch-local COMSOL parametric candidate.
- The remaining issue is not whether center supervision matters; S158/S161/S181-S185 already show that it does.
- Seed2/seed3 keep strong test IoU but have lower val IoU than seed1, which points to center localization stability rather than a need for larger center-grid lambda.
- A signal-to-center auxiliary head directly strengthens the signal latent's center-localization information, while preserving the successful bin-offset representation.

## Not recommended now

- Do not continue tuning center lambda values.
- Do not continue `center_axis_relative`.
- Do not immediately generate COMSOL V3 data before exhausting the current representation diagnostic.
- Do not return to type/rotation loss sweeps.
- Do not return to raster or forward-consistency sweeps.
- Do not return to dense conditional mask runner.

## Suggested next-stage acceptance

The next stage should compare current `center_bin_offset_plus_grid` against the same model with a signal-to-center auxiliary head. It should pass only if val/test IoU and held-out center error improve without degrading presence or causing type/axis/rotation tradeoffs that erase mask gains.

## Self-review

S188 selects one route and avoids expanding experiment cost prematurely. The recommendation follows directly from S187: center-bin works, but center information in the signal latent remains unstable on val.
