# S338 Component-Query 1-Sample Current Reference

S338 复跑 S330 原配置作为 same-run reference：

- sample: train source sample `0`
- steps: `10000`
- seed: `1`
- hidden_dim / latent_dim: `128 / 64`
- output: `experiments/dual_network/S338_component_query_1sample_current_reference/current_reference`

| metric | value |
| --- | ---: |
| polygon_mask_iou | `0.974226804` |
| pred / target area | `194 / 189` |
| presence_acc | `1.000000` |
| present_type_acc | `1.000000` |
| center_x_bin_acc | `1.000000` |
| center_y_bin_acc | `1.000000` |
| decoded_vertex_mae | `5.918177e-06` |
| local_vertex_mae_grid | `9.718776e-03` |
| out_of_grid_vertex_count | `0` |
| signed_area_flip_count | `0` |

The reference exactly reproduces S330, so S339 repair runs are valid.
