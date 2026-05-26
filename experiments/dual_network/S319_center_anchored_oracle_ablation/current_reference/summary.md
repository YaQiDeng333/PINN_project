# Center-anchored oracle ablation: current_reference

- pred_all max exported IoU diff: `0.000000e+00`
- pred_all max exported pred_area diff: `0`

| split | variant | mean IoU | min IoU | zero-IoU | mean area abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| train | pred_all | `0.995598` | `0.969697` | `0` | `0.633333` |
| train | gt_center_bin | `0.995598` | `0.969697` | `0` | `0.633333` |
| train | gt_offset | `0.998390` | `0.986667` | `0` | `0.200000` |
| train | gt_center_bin_offset | `0.998390` | `0.986667` | `0` | `0.200000` |
| train | gt_local | `0.998046` | `0.989130` | `0` | `0.300000` |
| train | gt_center_bin_offset_local | `1.000000` | `1.000000` | `0` | `0.000000` |
| val | pred_all | `0.037245` | `0.000000` | `8` | `91.300000` |
| val | gt_center_bin | `0.273894` | `0.033333` | `0` | `91.300000` |
| val | gt_offset | `0.056346` | `0.000000` | `7` | `93.100000` |
| val | gt_center_bin_offset | `0.450778` | `0.102679` | `0` | `93.100000` |
| val | gt_local | `0.058471` | `0.000000` | `7` | `17.900000` |
| val | gt_center_bin_offset_local | `1.000000` | `1.000000` | `0` | `0.000000` |
| test | pred_all | `0.072368` | `0.000000` | `9` | `74.200000` |
| test | gt_center_bin | `0.346678` | `0.000000` | `1` | `74.200000` |
| test | gt_offset | `0.075000` | `0.000000` | `9` | `78.200000` |
| test | gt_center_bin_offset | `0.438502` | `0.066667` | `0` | `78.200000` |
| test | gt_local | `0.095985` | `0.000000` | `8` | `15.900000` |
| test | gt_center_bin_offset_local | `1.000000` | `1.000000` | `0` | `0.000000` |
