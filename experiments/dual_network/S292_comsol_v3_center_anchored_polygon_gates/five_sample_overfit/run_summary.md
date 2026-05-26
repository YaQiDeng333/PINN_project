# COMSOL center-anchored polygon inverse run summary

## Config

- steps: `10000`
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
- export_predictions: `True`
- seed: `1`

## Final Metrics

| split | polygon_mask_iou | polygon_mask_iou_min | decoded_vertex_mae | presence_acc | type_acc | x_bin_acc | y_bin_acc | out_of_grid | signed_flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.991549` | `0.957746` | `1.353233e-05` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0` | `0` |
| val | `0.991549` | `0.957746` | `1.353233e-05` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0` | `0` |
| test | `0.991549` | `0.957746` | `1.353233e-05` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0` | `0` |

No checkpoint or model weights are saved by this runner.
