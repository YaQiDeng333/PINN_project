# Center decode diagnostics: current_reference

| split | mean_iou | zero_iou | x_bin_acc | y_bin_acc | hard_center_l2 | soft_center_l2 | local_vertex_mae_grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.995598` | `0` | `1.000000` | `1.000000` | `0.028688` | `0.028689` | `0.009393` |
| val | `0.037245` | `8` | `0.230769` | `0.230769` | `21.748188` | `19.000010` | `3.674866` |
| test | `0.072368` | `9` | `0.583333` | `0.083333` | `15.275538` | `13.161899` | `2.970076` |

Soft expected-center fields are `nan` when the prediction export predates S323 support.
