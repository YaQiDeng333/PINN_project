# S241 COMSOL V3 repaired candidate evaluation decision

## S238 Zero-Shot

V2-train to repaired-V3 val/test zero-shot did not produce metrics because the repaired V3 data remains in raw COMSOL coordinates and trips the train-grid center-bin range check:

```text
ValueError: center_x target is outside the x grid range.
```

This is a coordinate-convention/runability issue for cross-grid zero-shot, not a repaired-signal failure.

## S239 Repaired V3 Train Probe

The repaired V3 train split is learnable. The current candidate reaches train IoU `0.998851`, while param-only reaches `0.986927`. Held-out results are still weak:

- candidate val/test IoU: `0.052874` / `0.197143`
- param-only val/test IoU: `0.000000` / `0.157851`

The candidate is not robust enough on repaired V3 held-out splits to claim it handles this fallback pilot, although the repaired signal route makes train fitting possible.

## S240 Grouped Diagnostics

Grouped diagnostics show failure is concentrated in held-out center localization, especially y-bin/generalization:

- candidate val: only `both_bins_wrong_like` has nonzero mean IoU (`0.264368`); other groups are `0.000000`.
- candidate test: `bins_correct_center_or_offset_bad` is highest (`0.349162`), while `geometry_or_type_interaction` and `rare_y_bin_wrong` are `0.000000`.
- candidate train: all groups are near-perfect (`0.993103` to `1.000000`).

## Decision

The current S185 candidate is not validated as a repaired V3 fallback candidate. It can fit repaired V3 train, but held-out val/test are too weak and too split-sensitive.

Next unique recommendation: generate a larger repaired V3 hard-case pack with the same repaired Bz route before running mixed V2+V3 training or multi-seed candidate validation. The current `30/10/10` fallback pilot is sufficient to prove the signal route and train learnability, but not sufficient to support a stable candidate decision.

Boundary: this is still branch-local evidence and not a main baseline replacement.
