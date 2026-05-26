# S198 COMSOL x-bin failure stage summary

## Context

S194-S197 showed that the current S185 `center_bin_offset_plus_grid` candidate still has residual center-bin failures.

Key S196 observations:

- current candidate val/test x wrong rate = `0.200000` / `0.133333`;
- current candidate val/test y wrong rate = `0.083333` / `0.033333`;
- slots 0 and 2 are more fragile than slot 1;
- low-IoU samples usually have higher center-grid error;
- signal-to-center auxiliary head did not improve held-out IoU or center-grid error.

## Stage Direction

S198-S202 tests a narrow x-bin calibration route:

- keep the main S185 candidate as the baseline;
- do not modify model structure;
- do not enable the auxiliary center head;
- do not use dynamic hard-sample mining;
- only add optional weights to the main center-bin CE loss.

The S199 implementation adds `--center-bin-x-weight`, `--center-bin-y-weight`, and `--center-bin-slot-weights`. Defaults preserve the old behavior.

## Self-Review

- This stage targets the S196 x-bin bottleneck directly.
- It avoids using val/test worst samples as training weights.
- It keeps S185 as the current branch candidate until a future multi-seed validation proves a replacement.
