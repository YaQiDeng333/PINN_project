# S187 COMSOL center-bin error diagnostics

## Inputs

S187 reads existing S179 / S183 / S184 / S185 artifacts only. No new training was run. Prediction CSV files were available, so S187 reconstructs per-axis center bins from the train x/y grid and the S177 bin rule.

## Aggregate stability

| split | val/test IoU range | center_grid_mae range | x_bin_acc range | y_bin_acc range |
|---|---|---|---|---|
| val | 0.484303-0.542935 | 3.362513-6.282760 | 0.716667-0.800000 | 0.800000-0.883333 |
| test | 0.575504-0.581320 | 2.721649-2.929023 | 0.833333-0.850000 | 0.900000-0.950000 |

## Sample-level center error correlation

| split | Pearson(mean center L2 grid error, mask IoU) | Spearman |
|---|---:|---:|
| val | -0.773935 | -0.944914 |
| test | -0.910810 | -0.869630 |

## Main diagnostics

1. Test is relatively stable because all three seeds keep test center_grid_mae near `2.72-2.93`, with test x/y bin accuracy at or above `0.833333` / `0.900000`. Test IoU remains tightly grouped at `0.575504-0.581320`.
2. Val fluctuates more because seed2/seed3 have much higher val center_grid_mae (`6.282760` / `6.026593`) than seed1 (`3.362513`). The aggregate val IoU drop aligns with weaker val center localization.
3. The likely bottleneck is x-bin stability first, with y-bin secondary: val x-bin accuracy drops to `0.716667` in seed2 while y-bin accuracy is `0.800000`; seed1 is stronger on both axes but especially y (`0.883333`).
4. Existing aggregate metrics explain much of the val gap through center error, but not all sample-level IoU variation. Per-sample correlations are modest to moderate, so type/axis/rotation interaction and sample geometry still matter.
5. The next route should be signal-to-center auxiliary head rather than more bin/lambda tuning. The center-bin representation works, but val instability suggests the shared signal latent should receive a more direct center-localization training signal.

## Artifacts

- `center_bin_stability_table.csv`: run-level train/val/test metrics.
- `per_component_center_bin_errors.csv`: reconstructed per-component bin correctness and center error.
- `sample_center_bin_error_summary.csv`: per-sample center-bin / center error summary joined with mask IoU.
- `grouped_center_bin_errors.csv`: slot-level bin/error aggregates for val/test.
- `worst_val_samples.csv`: lowest-IoU val samples across seeds.

## Self-review

The diagnostics use only existing outputs. Reconstructed bin correctness matches the aggregate pattern in metrics, and no contradiction with S184/S185 was found.
