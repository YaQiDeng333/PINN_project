# S319 center-anchored oracle ablation summary

Offline ablation replayed S316 prediction exports and re-rasterized fixed variants without training or modifying the runner. `pred_all` exactly matched exported mask metrics for both runs: max IoU diff `0.000000e+00`, pred-area diff `0`.

## Current reference

| split | pred_all | gt_center_bin | gt_center_bin_offset | gt_local | gt_all |
| --- | ---: | ---: | ---: | ---: | ---: |
| val IoU / zero | `0.037245 / 8` | `0.273894 / 0` | `0.450778 / 0` | `0.058471 / 7` | `1.000000 / 0` |
| test IoU / zero | `0.072368 / 9` | `0.346678 / 1` | `0.438502 / 0` | `0.095985 / 8` | `1.000000 / 0` |

## Conditioning center-bin run

| split | pred_all | gt_center_bin | gt_center_bin_offset | gt_local | gt_all |
| --- | ---: | ---: | ---: | ---: | ---: |
| val IoU / zero | `0.027215 / 9` | `0.270234 / 1` | `0.495734 / 0` | `0.034444 / 8` | `1.000000 / 0` |
| test IoU / zero | `0.067059 / 9` | `0.408424 / 0` | `0.574024 / 0` | `0.055102 / 9` | `1.000000 / 0` |

Decision: center decode is the dominant held-out bottleneck. Replacing center bins or full center restores both splits far more than replacing local vertices alone, and `gt_center_bin_offset_local` reaches oracle IoU `1.0`. This passes the S319 center-main-cause gate and allows exactly one joint center/local repair quick gate.
