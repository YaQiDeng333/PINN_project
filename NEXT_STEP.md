NEXT_STEP

## Current Status

`CURRENT_BASELINE` is still the v3_complex mask-only grid decoder boundary model with validation-selected probability threshold `0.90`.

The project target is defect boundary shape inversion. Baseline decisions should prioritize IoU, Dice, area_error, `pred_area=0`, small / low-signal behavior, and whether the predicted mask explains the observed Bz signal.

The most valuable pending candidate is forward consistency with `lambda_forward=0.10`. Step 18.4 shows clear positive signal versus the current mask-only grid decoder baseline:

* IoU and Dice improve.
* area_error decreases.
* center_error improves.
* Bz residual / Bz MSE decreases strongly.
* `pred_area=0` does not clearly worsen.
* small / low-signal IoU and Dice improve, although area_error still needs review.

This candidate has not yet replaced `CURRENT_BASELINE`. Baseline replacement should wait for review / decision.

## Immediate Next Step

Prioritize Step 18.4 forward consistency `lambda_forward=0.10`.

Review should check:

* mask-to-Bz surrogate reliability and whether it was frozen correctly;
* checkpoint selection source and whether validation-only selection was preserved;
* probability threshold selection and whether test set was used only for final evaluation;
* metrics CSV / summary consistency;
* Bz residual calculation and comparison to current baseline;
* small / low-signal and polygon / rotated_rect trade-offs.

If review passes, discuss whether forward consistency `lambda_forward=0.10` should be promoted to a new `CURRENT_BASELINE`.

If forward consistency cannot replace the baseline, move to a new geometry-aware / physics-consistent inversion phase rather than continuing small decoder or threshold edits.

## Do Not Continue

Do not continue selection metric tuning, ensemble variants, threshold tricks, loss tricks, or small decoder patches.

Do not continue SDF v2, boundary head v2, coordinate refinement v2, hand-crafted Bz features, ordinary U-Net-like decoder variants, shape-type conditional variants, star-convex variants, retrieval variants, or other small fixes around already-stopped directions.
