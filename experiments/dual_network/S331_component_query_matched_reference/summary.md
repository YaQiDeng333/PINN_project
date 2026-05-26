# S331 Component-Query Matched Split Same-Run Reference

S331 was skipped.

## Reason

S330 one-sample gate failed with polygon IoU `0.974227`, below the required `>=0.99`. The stage plan requires stopping before 5-sample, same-run reference, or train30 when the 1-sample gate fails.

## Not Run

- No S299 matched split center-anchored reference rerun.
- No component-query train30 quick gate.
- No multi-seed.
- No new COMSOL data.

## Boundary

This skipped stage preserves the gate semantics. It avoids comparing held-out results from a component-query route that has not proven 1-sample raster precision.
