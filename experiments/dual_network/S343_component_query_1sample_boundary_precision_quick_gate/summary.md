# S343 Component-Query 1-Sample Boundary Precision Quick Gate

S343 runs only three 1-sample gates on train sample `0`.

| run | IoU | area | FP/FN | status |
| --- | ---: | ---: | ---: | --- |
| current_reference | `0.974226804` | `194 / 189` | `5 / 0` | reference |
| center_aux_half | `0.989528796` | `191 / 189` | `2 / 0` | failed |
| center_aux_half_plus_tiny_area | `0.979166667` | `191 / 189` | `3 / 1` | failed |

`center_aux_half` is the closest result. It reduces symmetric diff from `5` pixels to `2` pixels and keeps area gap at `+2`, but it still misses the explicit hard IoU gate `>=0.99`. Adding tiny area aux does not help.

By stop condition, no 5-sample or train30 run is allowed from this stage.
