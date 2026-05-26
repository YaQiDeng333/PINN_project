# S302 Polygon Matched-Coverage Decision

S298-S302 tested whether the center-anchored polygon held-out failure is mostly caused by the original train/val/test center-bin coverage gap.

## Result

S299 successfully built a diagnostic resplit from the existing 50 polygon V3 samples. Hard-case counts remain at the target distribution: train `10/5/7/5/3`, val `3/2/2/2/1`, and test `3/2/2/2/1`. All `20/20` held-out samples have every component bin within train center-bin distance `<=1`, but only `4/20` are exactly covered by train.

S300 train fit remains strong: train mean/min IoU is `0.995598` / `0.969697`, with presence/type and x/y bin accuracy all `1.000000`. Held-out performance does not improve: val/test mean IoU is `0.037245` / `0.072368`, compared with original `0.072402` / `0.084416`; zero-IoU is `8/10` and `9/10`, compared with original `8/10` and `8/10`.

S301 shows the remaining matched-coverage failure is still center-bin dominated, especially y-bin. Held-out zero-IoU is `17/20`; all `17/17` zero-IoU samples have y-bin errors, and `9/17` have x-bin errors. True rotated and multi-component samples remain difficult, but the dominant mechanism is still y-bin/localization failure.

## Decision

The matched-coverage gate does not support "original coverage gap alone explains the held-out failure." Distance-1 coverage is sufficient for the split diagnostic but not sufficient for the current center-anchored model to generalize.

The next unique recommendation is center-anchored y-bin localization repair, not multi-seed validation, more steps, larger model, or larger COMSOL data. A stricter exact-bin split can be used only as a small diagnostic control if needed, because this resplit still has exact coverage for only `4/20` held-out samples.

The S185/S181 center-bin candidate remains unchanged, and this resplit result is not a main baseline replacement.
