# S286 COMSOL V3 Polygon Prediction Failure Diagnostics

S286 diagnoses the `longer_train30` prediction exports from S282. No training is run.

## Outputs

- `prediction_failure_per_sample.csv`
- `prediction_failure_per_component.csv`
- `grouped_prediction_failures.csv`
- `worst_val_test_polygon_samples.csv`

## Findings

The held-out failure is dominated by vertex / shape prediction, not by presence/type alone.

| split | zero-IoU samples | mean vertex MAE | presence-bad samples | type-bad samples | out-of-grid vertices | signed-area flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0/30` | `5.575231e-05` | `0` | `0` | `0` | `0` |
| val | `4/10` | `5.368527e-03` | `2` | `3` | `4` | `7` |
| test | `6/10` | `3.298867e-03` | `1` | `1` | `3` | `5` |

Val/test have many samples where presence/type are correct but IoU is still zero. That points to vertex/shape extrapolation failure rather than classification failure.

Worst examples:

- val sample `1`, `x_bin_wrong_like`: IoU `0.000000`, target/pred area `210` / `152`, vertex MAE `5.875361e-03`.
- val sample `7`, `geometry_or_type_interaction`: IoU `0.000000`, target/pred area `105` / `235`, vertex MAE `1.246909e-02`, `3` out-of-grid vertices, `2` signed-area flips.
- test sample `5`, `bins_correct_center_or_offset_bad`: IoU `0.000000`, target/pred area `61` / `193`, vertex MAE `3.655933e-03`, `1` signed-area flip.

## Interpretation

The failure is broad, not a small number of outliers: val has `4/10` zero-IoU samples and test has `6/10`. The most important pathology is held-out vertex/shape instability, including signed-area flips, occasional out-of-grid vertices, and large area errors.
