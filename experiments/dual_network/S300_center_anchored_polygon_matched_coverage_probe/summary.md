# S300 Center-Anchored Polygon Matched-Coverage Probe

S300 trains the unchanged center-anchored polygon runner on the S299 matched-coverage resplit. This is a diagnostic run only; it does not replace S185/S181, does not enter multi-seed validation, and saves no weights or checkpoints.

## Configuration

- train/val/test samples: `30` / `10` / `10`.
- steps: `20000`.
- seed: `1`.
- hidden_dim / latent_dim: `128` / `64`.
- export predictions: `true`.

## Metrics

| split | mean IoU | min IoU | zero-IoU | x-bin acc | y-bin acc | presence acc | type acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.995598` | `0.969697` | `0/30` | `1.000000` | `1.000000` | `1.000000` | `1.000000` |
| val | `0.037245` | `0.000000` | `8/10` | `0.230769` | `0.230769` | `0.966667` | `0.769231` |
| test | `0.072368` | `0.000000` | `9/10` | `0.583333` | `0.083333` | `0.933333` | `0.833333` |

## Gate Result

The train gate passes, but held-out metrics do not improve over the original S292 split. Original val/test mean IoU was `0.072402` / `0.084416` with zero-IoU `8/10` / `8/10`; matched coverage gives `0.037245` / `0.072368` with zero-IoU `8/10` / `9/10`.

This means distance-1 matched center-bin coverage is not enough to rescue held-out performance.
