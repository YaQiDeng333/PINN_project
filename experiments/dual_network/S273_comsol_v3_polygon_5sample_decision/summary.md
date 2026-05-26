# S273 COMSOL V3 Polygon 5-Sample Decision

S269-S273 completes the 5-sample polygon inverse overfit gate.

## Decision

The 5-sample gate passes. The polygon inverse model can fit a small mixed hard-case subset covering all five hard-case labels, including rotated and multi-component examples.

This does not promote a polygon inverse candidate and does not replace the S185/S181 center-bin branch candidate. It only clears the next staged training gate.

## Next Step

The next stage should plan a train30 / val10 / test10 quick probe on the S254-S258 polygon V3 pack. That stage should remain branch-local and should not be treated as a main baseline replacement.

If train30 later fits train but fails held-out val/test, the conclusion should be small-data generalization / architecture validation, not a failure of the polygon target schema, because S257 oracle and S271/S272 overfit gates now pass.
