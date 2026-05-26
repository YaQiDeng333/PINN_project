# COMSOL polygon inverse run summary

## Config

- steps: `10000`
- lr: `0.001`
- hidden_dim: `128`
- latent_dim: `64`
- max_components: `3`
- max_vertices: `4`
- type_vocab: `rectangular_notch, rotated_rect`
- lambda_presence: `1.0`
- lambda_type: `1.0`
- lambda_vertex: `50.0`
- lambda_center_aux: `0.0`
- lambda_box_aux: `0.0`
- lambda_area_aux: `0.0`
- lambda_edge_aux: `0.0`
- vertex_loss_space: `norm`
- vertex_smoothl1_beta: `0.005`
- export_predictions: `True`
- seed: `1`

## Final Metrics

| split | polygon_mask_iou | polygon_mask_iou_min | vertex_mae | presence_acc | present_type_acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | `0.731445` | `0.518519` | `1.793932e-04` | `1.000000` | `1.000000` |
| val | `0.033122` | `0.000000` | `9.746788e-03` | `0.933333` | `0.692308` |
| test | `0.089484` | `0.000000` | `5.539294e-03` | `0.966667` | `0.923077` |

No checkpoint or model weights are saved by this runner.
