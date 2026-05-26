# COMSOL polygon inverse run summary

## Config

- steps: `20000`
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
| train | `0.935101` | `0.802920` | `5.560893e-05` | `1.000000` | `1.000000` |
| val | `0.046352` | `0.000000` | `6.150539e-03` | `0.933333` | `0.769231` |
| test | `0.136720` | `0.000000` | `3.714611e-03` | `0.966667` | `0.923077` |

No checkpoint or model weights are saved by this runner.
