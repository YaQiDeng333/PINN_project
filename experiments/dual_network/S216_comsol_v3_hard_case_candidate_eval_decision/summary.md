# S216 COMSOL V3 hard-case candidate evaluation decision

S212-S215 evaluated the current S185/S181 center-bin candidate on the real COMSOL V3 hard-case fallback pilot. The stage did not modify model code, did not train dense runners, did not save weights, and did not claim main baseline replacement.

## S213 Zero-shot Result

The intended V2-train to V3 val/test zero-shot run did not produce valid metrics. It stopped before training because V3 center targets lie outside the V2 train grid used by the center-bin target builder.

- V2 train x/y range: `[-0.04,0.04]` / `[-0.01,0.01]`
- V3 val/test x/y range: `[0,4500]` / `[0,3000]`
- runner error: `ValueError: center_x target is outside the x grid range.`

This means current V3 fallback pack coordinates are not directly compatible with the V2-trained center-bin candidate.

## S214 V3 Small-train Result

| run | train IoU | val IoU | test IoU | val center_grid_mae | test center_grid_mae |
|---|---:|---:|---:|---:|---:|
| v3_train_candidate | 0.019715 | 0.046905 | 0.044968 | 37.474541 | 44.786293 |
| v3_train_param_only_reference | 0.038119 | 0.078177 | 0.036448 | 34.044182 | 37.399517 |

The current candidate does not handle the V3 hard-case pilot as-is. Even on V3 train, mask IoU remains near zero, so this is not a useful fine-tune result.

## S215 Grouped Diagnostics

The grouped diagnostics show broad failure rather than one clean hard-case class. `bins_correct_center_or_offset_bad` is consistently among the worst held-out groups, but other groups also have near-zero IoU. The immediate blocker is not hard-case label balancing; it is coordinate / representation compatibility and weak V3 learnability with this fallback pack.

## Decision

Do not proceed to larger V3 model training or candidate replacement from these results. The S185 `center_bin_offset_plus_grid` candidate remains the current branch candidate for V2-style data, but it is not validated on this V3 fallback pack.

## Next Unique Recommendation

Fix the V3 geometry coordinate convention before any further candidate evaluation: either regenerate the V3 hard-case pack with V2-compatible meter-scale `x/y/defect_params`, or add an explicit, documented V3-to-V2 geometry-unit conversion step before parametric target construction. After that, rerun the ingest/oracle gate and only then rerun zero-shot / small-train evaluation.
