# S290 COMSOL V3 Center-Anchored Polygon Targets

S290 builds center-anchored polygon targets from the S254/S256 polygon V3 pack.

`vertex_ordering` is normalized to canonical `clockwise_top_left`; the target builder also accepts the older alias `clockwise_start_min_y_then_min_x_in_normalized_space` for reproducibility with earlier polygon target packages.

| split | samples | mean decode IoU | min decode IoU | max decode abs error | center bins |
| --- | ---: | ---: | ---: | ---: | --- |
| train | `30` | `1.000000` | `1.000000` | `9.313225746e-10` | `25 x 13` |
| val | `10` | `1.000000` | `1.000000` | `2.328306437e-10` | `25 x 13` |
| test | `10` | `1.000000` | `1.000000` | `4.656612873e-10` | `25 x 13` |

The decode oracle confirms that center-bin + offset + local grid vertices reconstruct the original polygon vertices and masks without losing oracle capacity.
