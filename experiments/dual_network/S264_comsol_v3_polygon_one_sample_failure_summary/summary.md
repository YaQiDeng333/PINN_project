# S264 COMSOL V3 Polygon One-Sample Failure Summary

S264 records the S262 one-sample polygon inverse failure before any wider polygon training. The target and rasterizer are not the blocker: S257 polygon oracle IoU is still `1.000000`, and the S262 model predicts the correct component presence and type.

## S262 Result

| metric | value |
| --- | ---: |
| train polygon mask IoU | `0.883178` |
| train polygon Dice | `0.937965` |
| presence accuracy | `1.000000` |
| present type accuracy | `1.000000` |
| normalized vertex MAE | `4.207401e-05` |
| pred area | `214` |
| target area | `189` |

## Interpretation

The failure is most consistent with vertex-to-hard-raster sensitivity. The vertex regression error is numerically small in normalized coordinates, but the hard polygon rasterizer evaluates discrete grid pixels. A sub-cell edge shift can add or remove a boundary band of pixels and move IoU below the gate.

This stage does not relax the gate, does not enter 5-sample overfit, does not run train30, and does not replace the S185/S181 center-bin branch candidate. The next step is a one-sample raster-sensitivity diagnostic and a minimal one-sample repair probe.
