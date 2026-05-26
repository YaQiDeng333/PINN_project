# S311 bounded local output quick gate

Scope: matched split only, `steps=20000`, `seed=1`, export predictions enabled, no checkpoint or model weights saved.

## Results

| run | train IoU mean/min | val IoU mean/min | test IoU mean/min | val/test zero-IoU | val/test y-bin acc | val/test local MAE grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_reference | `0.995598 / 0.969697` | `0.037245 / 0.000000` | `0.072368 / 0.000000` | `8 / 9` | `0.230769 / 0.083333` | `3.674865 / 2.970076` |
| bounded_local_fixed_grid | `0.995925 / 0.961039` | `0.024490 / 0.000000` | `0.060554 / 0.000000` | `9 / 8` | `0.230769 / 0.166667` | `3.633342 / 2.862006` |
| bounded_local_train_stats | `0.989528 / 0.857143` | `0.029174 / 0.000000` | `0.067532 / 0.000000` | `7 / 9` | `0.230769 / 0.166667` | `3.447587 / 2.507451` |

## Gate Decision

The same-run `current_reference` reproduced the S300/S306 failure mode, so the quick gate is valid. Both bounded variants preserved train fit, but neither improved val and test IoU relative to the reference, and neither reduced zero-IoU on both held-out splits.

`bounded_local_train_stats` reduced val zero-IoU from `8/10` to `7/10` and lowered held-out local MAE, but test zero-IoU stayed `9/10` and mean IoU stayed below reference. This is a partial local-shape stabilization signal, not a pass.

The bounded output path did not introduce out-of-grid vertices or signed-area flips. Saturation remained `0.0`, so the current bounds are not clipping predictions; the failure is still dominated by center-bin/local-shape conditioning rather than raw local offset explosion.
