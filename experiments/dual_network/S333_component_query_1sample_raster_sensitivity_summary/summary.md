# S333 Component-Query 1-Sample Raster Sensitivity Summary

S333 starts an offline diagnostic stage for the S330 component-query one-sample gate failure. No training is run, no model or runner structure is changed, and the route does not enter 5-sample or train30.

## S330 Failure

- sample: train sample `0`
- hard_case_type: `x_bin_wrong_like`
- polygon IoU: `0.974227`
- required 1-sample gate: `>=0.99`
- presence/type/x-bin/y-bin accuracy: `1.000000`
- decoded vertex MAE: `5.918177e-06`
- pred / target raster area: `194` / `189`
- out-of-grid vertices: `0`
- signed-area flips: `0`

## Hypothesis

The failure is likely hard-raster sensitivity around a small boundary / centroid / area mismatch, not a component presence, type, center-bin classification, or target-rasterizer issue. The next step is to reconstruct the predicted polygon offline and test targeted sensitivity variants without changing the training path.

## Stop Boundary

Because the 1-sample gate failed, the 5-sample gate, same-run reference, and train30 gate remain skipped.
