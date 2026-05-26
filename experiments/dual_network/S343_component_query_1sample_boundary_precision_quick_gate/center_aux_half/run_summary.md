# COMSOL component-query polygon inverse run summary

This runner keeps the center-anchored target schema and hard argmax decode, but predicts all component outputs from shared fixed-slot query latents.

## Config

- inverse_route: `component_query`
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
- lambda_area_aux: `0.0`
- lambda_decoded_center_aux: `0.025`
- lambda_polygon_centroid_aux: `0.0`
- center_centroid_aux_smoothl1_beta: `0.01`
- export_predictions: `True`
- seed: `1`

## Final Metrics

| split | polygon_mask_iou | polygon_mask_iou_min | decoded_vertex_mae | local_vertex_mae_grid | hard_center_l2_grid | presence_acc | type_acc | x_bin_acc | y_bin_acc | zero_iou | out_of_grid | signed_flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.989529` | `0.989529` | `4.240013e-06` | `1.085255e-02` | `0.014208` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0` | `0` | `0` |
| val | `0.989529` | `0.989529` | `4.240013e-06` | `1.085255e-02` | `0.014208` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0` | `0` | `0` |
| test | `0.989529` | `0.989529` | `4.240013e-06` | `1.085255e-02` | `0.014208` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0` | `0` | `0` |

No checkpoint or model weights are saved by this runner.
