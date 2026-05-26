# S316 local-shape conditioning quick gate

Scope: S299 matched split, `steps=20000`, `seed=1`, export predictions enabled, no checkpoints or weights.

## Results

| run | train IoU mean/min | val IoU mean/min | test IoU mean/min | val/test zero-IoU | val/test y-bin acc | val/test local MAE grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_reference | `0.995598 / 0.969697` | `0.037245 / 0.000000` | `0.072368 / 0.000000` | `8 / 9` | `0.230769 / 0.083333` | `3.674865 / 2.970076` |
| conditioning_center_bin | `0.999749 / 0.992481` | `0.027215 / 0.000000` | `0.067059 / 0.000000` | `9 / 9` | `0.153846 / 0.166667` | `3.409837 / 2.449341` |
| conditioning_center_bin_slot | skipped | skipped | skipped | skipped | skipped | skipped |
| conditioning_center_bin_slot_type | skipped | skipped | skipped | skipped | skipped | skipped |

## Gate Decision

The same-run reference reproduced S311 exactly. `conditioning_center_bin` improved train fit and reduced held-out local MAE, but it reduced val/test IoU below reference and worsened val zero-IoU from `8/10` to `9/10`.

This triggers the S313-S317 stop condition: conditioning that only improves train or local MAE without improving held-out IoU does not pass. Slot and type variants were skipped to avoid expanding the matrix after the first mechanism failed.

No out-of-grid vertices or signed-area flips occurred.
