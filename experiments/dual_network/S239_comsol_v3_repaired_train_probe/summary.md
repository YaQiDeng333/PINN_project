# S239 COMSOL V3 repaired train quick probe

S239 trains on repaired V3 train and evaluates repaired V3 val/test. This checks whether the repaired pack is learnable within one coordinate convention.

## Metrics

| run | split | mask_iou | center_grid_mae | x_bin_acc | y_bin_acc | center_offset_mae |
|---|---|---:|---:|---:|---:|---:|
| candidate | train | 0.998851 | 0.017465 | 1.000000 | 1.000000 | 0.001336 |
| candidate | val | 0.052874 | 14.233663 | 0.700000 | 0.100000 | 0.267174 |
| candidate | test | 0.197143 | 13.523300 | 0.900000 | 0.300000 | 0.218402 |
| param_only_reference | train | 0.986927 | 0.159811 | n/a | n/a | n/a |
| param_only_reference | val | 0.000000 | 15.983777 | n/a | n/a | n/a |
| param_only_reference | test | 0.157851 | 12.593512 | n/a | n/a | n/a |

## Interpretation

The repaired V3 signal is learnable on train: both candidate and param-only fit train masks strongly, and the candidate reaches nearly perfect train center-bin metrics. Held-out val/test remain weak, especially val. The candidate is better than param-only on test IoU and clearly better on train, but neither configuration is a reliable held-out solution for this small fallback pilot.
