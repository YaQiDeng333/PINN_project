# S291 COMSOL V3 Center-Anchored Polygon Runner

S291 adds an independent center-anchored polygon inverse model and runner. The old absolute-vertex polygon runner remains unchanged and serves as the S282 reference.

The model outputs presence logits, type logits, x/y center-bin logits, center offsets, and grid-cell local vertices. The runner uses supervised losses only: presence BCE, present type CE, x/y center-bin CE, center-offset SmoothL1, and local-vertex SmoothL1. Hard polygon rasterization is evaluation-only; no differentiable rasterizer is introduced.

Smoke tests cover target encode/decode, model forward/backward, runner execution, prediction export, and no checkpoint/weight output.
