# S238 COMSOL V3 repaired zero-shot evaluation

S238 attempted V2-train to repaired-V3 val/test zero-shot with the current `center_bin_offset_plus_grid` candidate.

## Result

The run failed before training/evaluation metrics were produced:

```text
ValueError: center_x target is outside the x grid range.
```

This is not the old near-constant signal problem. It is the known cross-grid geometry convention issue: V2 train uses meter-scale centered geometry, while repaired V3 currently remains in raw COMSOL coordinates `[0,4500] / [0,3000]`. The repaired signals are non-degenerate, but this zero-shot command cannot be interpreted until repaired V3 is normalized or the runner explicitly supports cross-grid center-bin targets.

## Metrics

- repaired V3 val mask_iou: not available
- repaired V3 test mask_iou: not available
- center_grid_mae / x-y bin accuracy: not available

Decision: continue S239 same-grid repaired V3 train quick probe, because train/val/test within repaired V3 share the same raw coordinate convention.
