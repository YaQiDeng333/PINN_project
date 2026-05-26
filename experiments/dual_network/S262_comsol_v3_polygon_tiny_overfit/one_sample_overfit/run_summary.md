# COMSOL polygon inverse run summary

## Config

- steps: `3000`
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
- vertex_smoothl1_beta: `0.005`
- export_predictions: `True`
- seed: `1`

## Final Metrics

| split | polygon_mask_iou | polygon_mask_iou_min | vertex_mae | presence_acc | present_type_acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | `0.883178` | `0.883178` | `4.207401e-05` | `1.000000` | `1.000000` |
| val | `0.883178` | `0.883178` | `4.207401e-05` | `1.000000` | `1.000000` |
| test | `0.883178` | `0.883178` | `4.207401e-05` | `1.000000` | `1.000000` |

No checkpoint or model weights are saved by this runner.
