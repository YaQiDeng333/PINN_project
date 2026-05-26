# S231 normalized V3 tiny-overfit decision

S231 summarizes S227-S230. The tiny-overfit route was intentionally stopped before training.

## Findings

- S227 found no target/mask/bbox alignment failure.
- `masks == (mu_maps < 500)` has zero mismatch pixels on train/val/test.
- bbox center error is within roughly half a grid cell on each split.
- S219 targets match normalized defect parameters to sub-nanometer numerical tolerance for x/y/axis fields.
- center-bin targets are in range and offsets remain within `[-0.5,0.5]`.
- The blocking issue is signal scale: train/val/test all have `std_floor_trigger_rate=1.0` under the runner `std < 1e-8` rule.

## Decision

S228, S229, and S230 were skipped. Running tiny-overfit with every sample below the signal normalization floor would not explain whether the model can learn V3 geometry from a meaningful Bz signal.

## Next Step

The next unique recommendation is to inspect and repair the COMSOL V3 Bz signal export path: field expression, probe height, magnetization / source scaling, lift-off extraction, and the interaction with the runner `per_sample_zscore` floor. Do not generate a larger V3 pack and do not change the model or runner until the signal-scale issue is resolved.

The S185 `center_bin_offset_plus_grid` candidate remains the current V2-style branch candidate. This stage does not replace main baselines and does not validate the candidate on normalized V3.
