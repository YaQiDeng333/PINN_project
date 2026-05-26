# S192 COMSOL signal-to-center auxiliary full confirm skipped

S192 was skipped because S191 did not produce an auxiliary configuration that passed the quick gate.

The same-round `current_candidate_reference` reached val/test IoU `0.546311` / `0.586546`. Both auxiliary groups were lower on both held-out splits:

- `aux_center_bin_offset`: val/test IoU `0.516648` / `0.567790`;
- `aux_center_bin_offset_xweighted`: val/test IoU `0.542723` / `0.580217`.

The auxiliary groups also worsened held-out `center_grid_mae` versus the same-round reference. Therefore the stage does not run a 3000-step full confirm, does not increase auxiliary weights, and does not promote auxiliary head behavior.

The current COMSOL parametric candidate remains S185 `center_bin_offset_plus_grid`.
