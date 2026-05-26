# S285 COMSOL V3 Polygon Geometry And Signal Distribution Diagnostics

S285 runs `comsol_polygon_generalization_diagnostics.py` on the S254 polygon V3 pack and S282 `longer_train30` predictions. No training is run.

## Outputs

- `geometry_signal_per_sample.csv`
- `split_geometry_signal_distribution.csv`
- `grouped_geometry_signal_distribution.csv`
- `summary.md`

The same diagnostic invocation also writes prediction-failure tables consumed by S286.

## Distribution Summary

The coarse split design is not obviously broken:

- hard-case distributions match the pack design;
- true rotated rates are train/val/test `0.700` / `0.700` / `0.800`;
- true multi-component rates are train/val/test `0.233` / `0.300` / `0.300`;
- signal std mean is train/val/test `2.124018e-06` / `2.266323e-06` / `1.900533e-06`;
- lift-off std ratio stays same-scale: train/val/test `0.970033` / `0.972890` / `0.989317`.

The clearest distribution caveat is sparse x-coverage. `center_x` mean is train/val/test `-0.001439` / `-0.002947` / `0.008342`, so test is right-shifted relative to train/val. Test also has weaker left-side vertex-x coverage: train min vertex-x reaches `-0.035142`, while test only reaches `-0.016364`.

## Interpretation

Val/test failure is not explained by signal scale or a missing hard-case class. It is more consistent with small-sample geometry coverage sparsity, especially in x-position and component layout, combined with the direct vertex predictor's weak held-out extrapolation.
