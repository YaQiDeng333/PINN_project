# COMSOL center-anchored polygon inverse run summary

## Config

- steps: `20000`
- lr: `0.001`
- hidden_dim: `128`
- latent_dim: `64`
- max_components: `3`
- max_vertices: `4`
- type_vocab: `rectangular_notch, rotated_rect`
- center_bin_size_cells: `8`
- center_x_bins: `25`
- center_y_bins: `13`
- lambda_presence: `1.0`
- lambda_type: `1.0`
- lambda_center_bin: `1.0`
- lambda_center_offset: `10.0`
- lambda_local_vertex: `1.0`
- lambda_center_aux: `0.0`
- lambda_box_aux: `0.0`
- lambda_area_aux: `0.0`
- lambda_edge_aux: `0.0`
- center_y_bin_extra_loss_mode: `none`
- lambda_center_y_bin_extra: `0.0`
- center_y_bin_neighbor_smoothing: `0.0`
- center_y_bin_distance_sigma: `0.75`
- local_shape_output_mode: `bounded_tanh`
- local_shape_bound_mode: `train_stats`
- local_shape_bound_x_grid: `21.834723949432373`
- local_shape_bound_y_grid: `4.747534394264221`
- local_shape_train_stats_margin: `1.25`
- export_predictions: `True`
- seed: `1`

## Final Metrics

| split | polygon_mask_iou | polygon_mask_iou_min | decoded_vertex_mae | local_vertex_mae_grid | presence_acc | type_acc | x_bin_acc | y_bin_acc | y_bin_abs_err | y_bin_within1 | out_of_grid | signed_flip | local_sat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.989528` | `0.857143` | `5.358264e-06` | `1.247530e-02` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `0` | `0` | `0.000000` |
| val | `0.029174` | `0.000000` | `3.965378e-03` | `3.447587e+00` | `0.966667` | `0.538462` | `0.384615` | `0.230769` | `1.846154` | `0.538462` | `0` | `0` | `0.000000` |
| test | `0.067532` | `0.000000` | `3.855667e-03` | `2.507451e+00` | `0.933333` | `0.833333` | `0.666667` | `0.166667` | `1.833333` | `0.500000` | `0` | `0` | `0.000000` |

No checkpoint or model weights are saved by this runner.
