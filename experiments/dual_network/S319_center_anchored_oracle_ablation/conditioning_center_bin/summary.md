# Center-anchored oracle ablation: conditioning_center_bin

- pred_all max exported IoU diff: `0.000000e+00`
- pred_all max exported pred_area diff: `0`

| split | variant | mean IoU | min IoU | zero-IoU | mean area abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| train | pred_all | `0.999749` | `0.992481` | `0` | `0.033333` |
| train | gt_center_bin | `0.999749` | `0.992481` | `0` | `0.033333` |
| train | gt_offset | `1.000000` | `1.000000` | `0` | `0.000000` |
| train | gt_center_bin_offset | `1.000000` | `1.000000` | `0` | `0.000000` |
| train | gt_local | `0.999749` | `0.992481` | `0` | `0.033333` |
| train | gt_center_bin_offset_local | `1.000000` | `1.000000` | `0` | `0.000000` |
| val | pred_all | `0.027215` | `0.000000` | `9` | `89.200000` |
| val | gt_center_bin | `0.270234` | `0.000000` | `1` | `89.200000` |
| val | gt_offset | `0.032749` | `0.000000` | `9` | `90.000000` |
| val | gt_center_bin_offset | `0.495734` | `0.126050` | `0` | `90.000000` |
| val | gt_local | `0.034444` | `0.000000` | `8` | `11.600000` |
| val | gt_center_bin_offset_local | `1.000000` | `1.000000` | `0` | `0.000000` |
| test | pred_all | `0.067059` | `0.000000` | `9` | `59.200000` |
| test | gt_center_bin | `0.408424` | `0.019737` | `0` | `59.200000` |
| test | gt_offset | `0.082143` | `0.000000` | `9` | `59.700000` |
| test | gt_center_bin_offset | `0.574024` | `0.191837` | `0` | `59.700000` |
| test | gt_local | `0.055102` | `0.000000` | `9` | `15.400000` |
| test | gt_center_bin_offset_local | `1.000000` | `1.000000` | `0` | `0.000000` |
