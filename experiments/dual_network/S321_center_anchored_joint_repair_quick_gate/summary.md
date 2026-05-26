# S321 joint center/local-shape repair quick gate

Scope: S299 matched split, `steps=20000`, `seed=1`, export predictions enabled, no checkpoints or weights.

## Results

| run | train IoU mean/min | val IoU mean/min | test IoU mean/min | val/test zero-IoU | val/test y-bin acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_reference | `0.995598 / 0.969697` | `0.037245 / 0.000000` | `0.072368 / 0.000000` | `8 / 9` | `0.230769 / 0.083333` |
| soft_center_scheduled | `0.977544 / 0.818182` | `0.000000 / 0.000000` | `0.090810 / 0.000000` | `10 / 7` | `0.076923 / 0.166667` |

Gate decision: failed.

Reasons:

- Train fit fell below the S321 threshold `train mean IoU >= 0.99`.
- Val IoU regressed to `0.000000`, and val zero-IoU worsened to `10/10`.
- Test IoU improved slightly, but one-split improvement is not enough and cannot override train-fit/val failure.

No second repair variant was run.
