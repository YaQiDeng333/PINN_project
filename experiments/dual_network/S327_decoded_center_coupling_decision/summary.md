# S327 decoded-center coupling decision

S323-S327 completed the loss-side decoded-center coupling gate.

Conclusions:

- The S321/S326 reference is reproducible.
- The S319 oracle ablation remains the strongest evidence: GT center bin and offset recover held-out IoU far more than GT local vertices.
- A simple differentiable soft decoded-center consistency loss is not sufficient. It improves train center error but breaks train fit and collapses val.
- Because the first loss-side repair fails, S326 does not proceed to `soft_decoded_vertex_consistency`.

Decision: stop this stage. The next unique recommendation is a stronger structural center-local route, such as a component-query center/shape head or another architecture that predicts center and local shape through a shared component representation. Do not continue loss-weight tuning, y-loss, local-conditioning, bound sweeps, teacher-forcing variants, multi-seed validation, extra steps, or new COMSOL data from this failed gate.

This stage does not replace S185/S181, does not promote a polygon inverse candidate, and is not a main baseline replacement.
