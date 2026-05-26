# S201 COMSOL x-bin center calibration full confirm skipped

S201 was skipped because S200 did not produce a configuration that passed the quick gate.

The same-round reference reached val/test IoU `0.546311` / `0.586546`.

- `x_bin_weighted` reached val/test IoU `0.545284` / `0.555791`.
- `x_bin_slot_weighted` reached val/test IoU `0.518272` / `0.543191`.

Both new configurations worsened held-out center-grid error and did not improve val/test IoU against the same-round reference. Therefore no 3000-step full confirm was run, and no candidate replacement is proposed.
