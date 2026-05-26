# S213 V2-trained candidate to V3 zero-shot summary

S213 attempted to train the current S185/S181 center-bin candidate on V2 train and evaluate it directly on V3 hard-case val/test. The run was stopped before training because the V3 center targets are outside the V2 train grid used to construct center-bin targets.

## Result

No valid zero-shot V3 metrics were produced.

## Blocking Error

The runner raised:

```text
ValueError: center_x target is outside the x grid range.
```

## Grid / Unit Evidence

| dataset | x range | y range | center_x range | center_y range |
|---|---:|---:|---:|---:|
| V2 train | `[-0.04, 0.04]` | `[-0.01, 0.01]` | `[-0.0202, 0.0202]` | `[-0.0057, 0.0050]` |
| V3 val | `[0, 4500]` | `[0, 3000]` | `[781.407, 3418.090]` | `[810.000, 1856.061]` |
| V3 test | `[0, 4500]` | `[0, 3000]` | `[1066.332, 3770.402]` | `[1200.000, 2340.909]` |

The current center-bin candidate assumes the train split grid defines the bin coordinate system. V2 train and V3 hard-case pack are therefore not directly zero-shot compatible in center-bin mode.

## Interpretation

This is a geometry-unit / grid-coordinate compatibility issue, not a model performance result. The V3 pack passed its own ingest and oracle gates, but it is not coordinate-compatible with V2-trained center-bin targets without explicit unit harmonization or regenerated V3 coordinates in the V2 meter-scale frame.
