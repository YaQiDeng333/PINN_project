# S201 COMSOL x-bin center calibration full confirm summary

S201 was skipped.

Reason: neither S200 x-bin calibration configuration met the gate. In particular, `x_bin_weighted` reduced val x wrong rate but did not improve val/test IoU and worsened test x wrong rate; `x_bin_slot_weighted` degraded both held-out splits.

No full confirm was run. The current COMSOL parametric candidate remains S185 `center_bin_offset_plus_grid`.
