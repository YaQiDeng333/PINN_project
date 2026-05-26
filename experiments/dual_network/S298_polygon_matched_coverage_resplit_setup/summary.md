# S298 Polygon Matched-Coverage Resplit Setup

S294-S297 showed that center-anchored polygon held-out failure is strongly coupled to train center-bin coverage gaps. This stage therefore tests a matched-coverage resplit using the existing S254-S258 polygon V3 pack.

## Boundary

- This resplit is diagnostic only; it is not a final generalization benchmark.
- The original S254-S258 train/val/test data are not modified or overwritten.
- No new COMSOL data are generated.
- The model structure and `train_comsol_center_anchored_polygon_inverse.py` remain unchanged.
- The S185/S181 center-bin candidate remains unchanged and is not replaced.

## Goal

The goal is to distinguish coverage-gap failure from model/representation failure. If val/test improve under matched center-bin coverage, the original split is the main bottleneck. If matched coverage still fails, the next repair should target center-bin/local-shape generalization rather than data split design.
