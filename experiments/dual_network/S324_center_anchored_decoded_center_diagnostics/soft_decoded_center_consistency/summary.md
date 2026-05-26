# Center decode diagnostics: soft_decoded_center_consistency

| split | mean_iou | zero_iou | x_bin_acc | y_bin_acc | hard_center_l2 | soft_center_l2 | local_vertex_mae_grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.983633` | `0` | `1.000000` | `1.000000` | `0.017545` | `0.017564` | `0.011918` |
| val | `0.000000` | `10` | `0.384615` | `0.153846` | `23.569897` | `19.362015` | `3.330914` |
| test | `0.034211` | `9` | `0.916667` | `0.166667` | `12.706467` | `13.801410` | `3.097257` |

Soft expected-center fields are `nan` when the prediction export predates S323 support.
