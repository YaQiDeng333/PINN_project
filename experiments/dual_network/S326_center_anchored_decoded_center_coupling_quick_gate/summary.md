# S326 decoded-center coupling quick gate

Scope: S299 matched split, `steps=20000`, `seed=1`, export predictions enabled, no checkpoints or weights.

## Results

| run | train IoU mean/min | val IoU mean/min | test IoU mean/min | val/test zero-IoU | val/test y-bin acc | val/test hard center L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_reference | `0.995598 / 0.969697` | `0.037245 / 0.000000` | `0.072368 / 0.000000` | `8 / 9` | `0.230769 / 0.083333` | `21.748188 / 15.275539` |
| soft_decoded_center_consistency | `0.983633 / 0.857143` | `0.000000 / 0.000000` | `0.034211 / 0.000000` | `10 / 9` | `0.153846 / 0.166667` | `23.569897 / 12.706466` |

Gate decision: failed.

Reasons:

- Train mean IoU fell below the S326 threshold `0.99`.
- Val IoU collapsed to `0.000000` and zero-IoU worsened to `10/10`.
- Test IoU also fell below the reference.
- Although train center error improved, held-out center decode did not improve consistently.

`soft_decoded_vertex_consistency` was not run because the first repair group hit the stop condition.
