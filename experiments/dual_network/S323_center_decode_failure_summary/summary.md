# S323 center-decode failure summary

S323 starts a loss-side center decode repair after S318-S322. This stage does not run multi-seed, does not add steps, does not expand model capacity, does not generate new COMSOL data, and does not replace the S185/S181 center-bin candidate.

S319 shows the held-out mask failure is center-decode dominated:

| ablation | val IoU | test IoU | val/test zero-IoU |
| --- | ---: | ---: | ---: |
| pred_all | `0.037245` | `0.072368` | `8 / 9` |
| gt_center_bin | `0.273894` | `0.346678` | `0 / 1` |
| gt_offset | `0.056346` | `0.075000` | `7 / 9` |
| gt_center_bin_offset | `0.450778` | `0.438502` | `0 / 0` |
| gt_local | `0.058471` | `0.095985` | `7 / 8` |
| gt_center_bin_offset_local | `1.000000` | `1.000000` | `0 / 0` |

The main bottleneck is center-bin decode, especially y-bin. Offset alone is not enough to recover held-out masks. The S321 `soft_center_scheduled` route failed because it conditioned local shape on center context without directly improving the center-bin logits used by final hard decode.

S323-S327 therefore tests default-off differentiable decoded-center consistency losses. Hard argmax center decode remains the official eval/export path.
