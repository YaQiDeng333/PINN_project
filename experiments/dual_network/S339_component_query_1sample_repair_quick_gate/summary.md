# S339 Component-Query 1-Sample Repair Quick Gate

S339 only runs 1-sample repair variants after S338 reproduces the reference. It does not run 5-sample, train30, multi-seed, extra COMSOL data, or candidate replacement.

| run | IoU | pred / target area | status |
| --- | ---: | ---: | --- |
| current_reference | `0.974226804` | `194 / 189` | reference |
| decoded_center_aux_small | `0.984126984` | `186 / 189` | failed |
| polygon_centroid_aux_small | `0.963917526` | `192 / 189` | failed |
| center_plus_centroid_aux_small |  |  | skipped |

`decoded_center_aux_small` improves IoU but does not reach the required `>=0.99` gate. It also reverses the raster error direction: the S339 recheck reports false-positive / false-negative pixels `0 / 3`, compared with the S330/S334 reference `5 / 0`. `polygon_centroid_aux_small` is worse than reference.

The combined aux run is skipped by the plan condition because neither single-aux run passes and no single-aux run reaches the `>=0.985` conditional trigger. The 1-sample gate remains blocked.
