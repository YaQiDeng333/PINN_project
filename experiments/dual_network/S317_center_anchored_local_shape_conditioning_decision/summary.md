# S317 local-shape conditioning decision

S313-S317 completed the first local-shape conditioning gate without new COMSOL data, multi-seed validation, extra steps, model capacity expansion, or S185/S181 replacement.

Decision: **local-shape conditioning did not pass this gate**.

Evidence:

- default-off `none` mode reproduced S311 reference exactly;
- `conditioning_center_bin` kept train fit strong but did not improve held-out IoU;
- val/test local MAE improved, but final masks did not improve;
- val zero-IoU worsened from `8/10` to `9/10`;
- out-of-grid vertices and signed-area flips stayed `0`.

Mechanism update: feeding predicted center-bin context to local shape can make local coordinates numerically closer, but the final decoded mask still fails when center-bin localization is wrong. The next repair should treat center-bin and local-shape as a coupled prediction problem rather than adding more local-head context in isolation.

Next unique recommendation: plan a joint center-bin/local-shape repair, such as a consistency objective between decoded center and local polygon geometry or an architecture that predicts center and shape through a shared component query. Do not continue simple local conditioning variants, y-loss tuning, bound sweeps, multi-seed validation, or new COMSOL data yet.
