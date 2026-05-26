# S330 Component-Query Polygon Overfit Gates

S330 触发 stop condition：component-query 1-sample gate 未通过，因此 5-sample 和 train30 均跳过。

## One-Sample Gate

- sample: train source sample `0`
- hard_case_type: `x_bin_wrong_like`
- steps: `10000`
- seed: `1`
- output: `experiments/dual_network/S330_component_query_polygon_overfit_gates/one_sample_overfit`

| metric | value |
| --- | ---: |
| polygon_mask_iou | `0.974227` |
| polygon_mask_iou_min | `0.974227` |
| presence_acc | `1.000000` |
| present_type_acc | `1.000000` |
| center_x_bin_acc | `1.000000` |
| center_y_bin_acc | `1.000000` |
| decoded_vertex_mae | `5.918177e-06` |
| local_vertex_mae_grid | `9.718776e-03` |
| pred_area / target_area | `194 / 189` |
| out_of_grid_vertex_count | `0` |
| signed_area_flip_count | `0` |

Acceptance required IoU `>=0.99`; observed IoU was `0.974227`. The model learned presence/type/bin targets, but the final raster mask remained five pixels larger than target. This is close to the gate but still below the explicit threshold.

## Skipped Gates

- 5-sample gate: skipped because 1-sample failed.
- same-run reference: skipped because 1-sample failed.
- train30 quick gate: skipped because 1-sample failed.

## Interpretation

The new component-query path is runnable, but the current query head does not yet satisfy the strict 1-sample hard-raster precision gate. The next diagnostic should focus on component-query one-sample raster sensitivity and local/area precision, not on held-out generalization, multi-seed, extra steps, or train30.
