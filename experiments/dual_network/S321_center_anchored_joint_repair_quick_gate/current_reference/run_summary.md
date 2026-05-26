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

| split | polygon_mask_iou | polygon_mask_iou_min | decoded_vertex_mae | local_vertex_mae_grid | presence_acc | type_acc | x_bin_acc | y_bin_acc | y_bin_abs_err | y_bin_within1 | out_of_grid | signed_flip | local_sat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.995598` | `0.969697` | `6.483366e-06` | `9.392819e-03` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `0` | `0` | `0.000000` |
| val | `0.037245` | `0.000000` | `4.156881e-03` | `3.674865e+00` | `0.966667` | `0.769231` | `0.230769` | `0.230769` | `1.769231` | `0.461538` | `0` | `0` | `0.000000` |
| test | `0.072368` | `0.000000` | `2.632789e-03` | `2.970076e+00` | `0.933333` | `0.833333` | `0.583333` | `0.083333` | `1.916667` | `0.333333` | `0` | `0` | `0.000000` |

No checkpoint or model weights are saved by this runner.
