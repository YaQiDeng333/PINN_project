# S322 joint center/local-shape decision

S318-S322 completed the center/local causal diagnostic and one allowed quick repair.

Conclusions:

- S319 proves the held-out failure is center-decode dominated: GT center-bin or GT center-bin+offset restores val/test IoU by far more than GT local vertices.
- Simple local-shape conditioning failed in S316 because it made local coordinates closer but did not repair the center decode used by final masks.
- The first minimal joint repair, `soft_center_scheduled`, did not pass. It weakened train fit and collapsed val IoU, so this exact teacher-forced soft-center path should not be promoted.

Next unique recommendation: plan a more explicit center-local coupling route, such as a decoded-center consistency loss or component-query center/shape head, but only after a fresh plan. Do not run more ad hoc joint variants, local-conditioning sweeps, y-loss tuning, bound sweeps, multi-seed validation, or new COMSOL data from this stage.
