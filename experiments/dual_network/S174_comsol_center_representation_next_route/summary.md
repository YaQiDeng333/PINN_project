# S174 COMSOL center representation next route

## Decision

The next recommended COMSOL parametric route is `center-bin classification + offset`.

## Rationale

- S158 oracle ablation showed center localization is the dominant mask IoU bottleneck.
- S161 showed val/test center error is strongly negatively correlated with mask IoU.
- S163/S164 showed center-grid regression loss improves the current model.
- S166-S170 showed `lambda_center_grid=0.1` is stable across the available repeat runs.
- Further simple lambda tuning is lower information than changing the center representation.

## Deferred options

- `signal-to-center auxiliary head` remains a second-choice route if center-bin classification + offset fails.
- `per-component peak-position alignment` remains a later diagnostic feature route.
- COMSOL V3 data design should be prepared after the next center representation probe, unless the center-bin route exposes a clear data coverage failure.

## Stop condition for the next route

If center-bin classification + offset does not preserve or improve val/test mask IoU against the current center-grid candidate, or if it lowers center error without improving mask IoU, stop the representation route and reconsider data design or a signal-to-center auxiliary head.

## Self-review

S174 selects exactly one next direction and avoids returning to type, rotation, forward consistency, raster loss, or center lambda sweeps.
