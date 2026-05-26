# S197 COMSOL center-bin failure decision

## Decision

The next unique direction is:

**center-x-bin focused calibration / hard-sample refinement of the main center-bin output.**

## Rationale

S196 shows the remaining held-out failure is more x-bin dominated than y-bin dominated:

- current candidate val x/y wrong rates: `0.200000` / `0.083333`;
- current candidate test x/y wrong rates: `0.133333` / `0.033333`;
- slots 0 and 2 are more x-bin fragile than slot 1;
- low-IoU samples have substantially higher center-grid error and bin wrong rates.

S191/S196 also show that the current auxiliary-head route is not effective:

- plain auxiliary head worsened x/y wrong rates and IoU;
- x-weighted auxiliary head only slightly reduced val x wrong rate but worsened y wrong rate and center-grid error;
- neither auxiliary variant improved val/test IoU versus the same-round reference.

## Recommended Next Stage

Do not continue auxiliary-head lambda tuning. Do not return to raster, forward consistency, type/rotation loss, or dense conditional mask runner work.

The next stage should diagnose and design one focused intervention around the main center-bin output:

- x-bin class / hard-sample calibration;
- x-bin error weighting for slots 0 and 2;
- sample-level hard-case balancing for low-IoU, high-center-error validation-like patterns.

This should first be planned as a diagnostic / quick-gate stage. It should keep S185 `center_bin_offset_plus_grid` as the current COMSOL parametric candidate unless a future multi-seed validation proves a replacement.

## Boundary

The current candidate remains S185 `center_bin_offset_plus_grid` on `feature/dual-network-variational`. It is still branch-local and not a main baseline replacement.
