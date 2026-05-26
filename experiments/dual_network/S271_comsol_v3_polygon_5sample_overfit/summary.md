# S271 COMSOL V3 Polygon 5-Sample Overfit

S271 runs the 5-sample polygon inverse overfit gate with the S267 successful longer-overfit configuration. The same subset is used for train, val, and test so the result only measures train-fit capacity.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| mean train polygon IoU | `0.996028` |
| min train polygon IoU | `0.985401` |
| mean train polygon Dice | `0.998002` |
| presence accuracy | `1.000000` |
| present type accuracy | `1.000000` |
| normalized vertex MAE | `5.359486e-06` |
| mean pred area | `137.4` |
| mean target area | `137.6` |

## Gate

The run passes the 5-sample gate:

- mean IoU `>=0.95`
- min IoU `>=0.90`
- presence/type accuracy `=1.0`
- no degenerate component count or area collapse

No train30 quick probe is run in this stage.
