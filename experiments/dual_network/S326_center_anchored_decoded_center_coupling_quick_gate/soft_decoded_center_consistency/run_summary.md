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
- center_consistency_mode: `soft_decoded_center`
- lambda_center_consistency: `1.0`
- center_consistency_smoothl1_beta: `0.1`
- center_y_bin_extra_loss_mode: `none`
- lambda_center_y_bin_extra: `0.0`
- center_y_bin_neighbor_smoothing: `0.0`
- center_y_bin_distance_sigma: `0.75`
- local_shape_output_mode: `raw`
- local_shape_bound_mode: `fixed_grid`
- local_shape_bound_x_grid: `24.0`
- local_shape_bound_y_grid: `8.0`
- local_shape_train_stats_margin: `1.25`
- local_shape_conditioning_mode: `none`
- local_shape_conditioning_dim: `16`
- joint_center_shape_mode: `none`
- joint_center_teacher_forcing_start: `1.0`
- joint_center_teacher_forcing_end: `0.0`
- joint_center_teacher_forcing_steps: `20000`
- export_predictions: `True`
- seed: `1`

## Final Metrics

| split | polygon_mask_iou | polygon_mask_iou_min | decoded_vertex_mae | local_vertex_mae_grid | hard_center_mae_grid | soft_center_mae_grid | presence_acc | type_acc | x_bin_acc | y_bin_acc | y_bin_abs_err | y_bin_within1 | out_of_grid | signed_flip | local_sat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.983633` | `0.857143` | `5.515406e-06` | `1.191818e-02` | `0.010650` | `0.010639` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `0` | `0` | `0.000000` |
| val | `0.000000` | `0.000000` | `4.386088e-03` | `3.330914e+00` | `13.331527` | `11.110701` | `0.966667` | `0.692308` | `0.384615` | `0.153846` | `1.846154` | `0.538462` | `0` | `0` | `0.000000` |
| test | `0.034211` | `0.000000` | `2.398308e-03` | `3.097257e+00` | `7.342355` | `8.517584` | `0.900000` | `0.833333` | `0.916667` | `0.166667` | `1.583333` | `0.500000` | `0` | `0` | `0.000000` |

No checkpoint or model weights are saved by this runner.
