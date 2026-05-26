# S324 decoded-center diagnostics summary

S324 adds `center_anchored_center_decode_diagnostics.py` and prediction-export fields for center decode analysis. The diagnostics join prediction CSVs with S299 matched-split targets and report hard argmax center error, soft expected-center error, bin errors, offset errors, local vertex error, and zero-IoU linkage.

## Same-run reference

| split | IoU mean | zero-IoU | x-bin acc | y-bin acc | hard center L2 grid | soft center L2 grid | local vertex MAE grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.995598` | `0` | `1.000000` | `1.000000` | `0.028688` | `0.028689` | `0.009393` |
| val | `0.037245` | `8` | `0.230769` | `0.230769` | `21.748188` | `19.000010` | `3.674866` |
| test | `0.072368` | `9` | `0.583333` | `0.083333` | `15.275538` | `13.161899` | `2.970076` |

The reference exactly reproduces the S321/S326 failure and confirms that held-out y-bin localization is still very weak.

## Soft decoded-center consistency

| split | IoU mean | zero-IoU | x-bin acc | y-bin acc | hard center L2 grid | soft center L2 grid | local vertex MAE grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `0.983633` | `0` | `1.000000` | `1.000000` | `0.017545` | `0.017564` | `0.011918` |
| val | `0.000000` | `10` | `0.384615` | `0.153846` | `23.569897` | `19.362015` | `3.330914` |
| test | `0.034211` | `9` | `0.916667` | `0.166667` | `12.706467` | `13.801410` | `3.097257` |

The new loss improves train center error but breaks the train-fit gate and collapses val IoU. It does not provide a stable held-out repair.
