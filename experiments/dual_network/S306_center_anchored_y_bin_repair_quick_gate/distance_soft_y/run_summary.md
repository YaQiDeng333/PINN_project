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
- center_y_bin_extra_loss_mode: `distance_soft_ce`
- lambda_center_y_bin_extra: `0.5`
- center_y_bin_neighbor_smoothing: `0.0`
- center_y_bin_distance_sigma: `0.75`
- export_predictions: `True`
- seed: `1`

## Final Metrics

| split | polygon_mask_iou | polygon_mask_iou_min | decoded_vertex_mae | presence_acc | type_acc | x_bin_acc | y_bin_acc | y_bin_abs_err | y_bin_within1 | out_of_grid | signed_flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.991753` | `0.925000` | `7.424057e-06` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `0` | `0` |
| val | `0.017919` | `0.000000` | `3.884848e-03` | `0.966667` | `0.615385` | `0.230769` | `0.153846` | `1.692308` | `0.538462` | `0` | `0` |
| test | `0.068354` | `0.000000` | `2.786921e-03` | `0.900000` | `0.833333` | `0.666667` | `0.083333` | `2.000000` | `0.333333` | `0` | `0` |

No checkpoint or model weights are saved by this runner.
