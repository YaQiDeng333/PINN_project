# S288 COMSOL V3 Polygon Generalization Decision

S284-S288 completes a no-training diagnostic pass for the polygon inverse held-out failure.

## Decision

Do not enter multi-seed validation yet. Do not increase steps, enlarge the model, or generate more data in this stage.

The next stage should address held-out vertex/shape stability. The recommended first branch is an output-shape / vertex-parameterization repair plan, because S286 shows signed-area flips, area blow-up/under-shoot, and occasional out-of-grid vertices on val/test. A controlled resplit or matched-val probe is also useful, but the prediction pathology should be handled before treating this as a candidate validation problem.

## Next-Stage Options By Diagnosis

- If prioritizing shape pathology: design bounded or residual vertex parameterization, plus area/box/edge regularization gate.
- If prioritizing split coverage: design controlled resplit / matched-val probe from the existing `30/10/10` pack.
- If prioritizing small-N memorization: design regularization or early-stop gate; do not just continue steps.
- If target/oracle inconsistency appears later: return to polygon target/export/rasterizer repair before training.

## Boundary

The S185/S181 center-bin candidate remains unchanged. The polygon inverse route is still branch-local and not a main baseline replacement.
